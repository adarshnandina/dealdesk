"""FastAPI service.

Endpoints are synchronous `def` functions; FastAPI runs them in a thread pool, which suits
the synchronous SQLAlchemy and LangGraph calls underneath.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, status
from langchain_core.messages import AIMessage
from sqlalchemy import text

from dealdesk.agent.outputs import DealSummary
from dealdesk.agent.service import (
    AgentService,
    NoPendingApprovalError,
    TurnResult,
    build_service,
    postgres_checkpointer,
)
from dealdesk.api.schemas import (
    ChatRequest,
    DealStageChangedEvent,
    DecisionRequest,
    HealthResponse,
    MessageView,
    PendingAction,
    ThreadResponse,
    TurnResponse,
    WebhookAccepted,
)
from dealdesk.config import Settings, get_settings
from dealdesk.crm.repository import CRMRepository, NotFoundError, ValidationError
from dealdesk.db.models import Source
from dealdesk.db.session import session_scope
from dealdesk.logging_config import configure_logging

logger = logging.getLogger(__name__)


def get_service(request: Request) -> AgentService:
    svc: AgentService = request.app.state.service
    return svc


Service = Annotated[AgentService, Depends(get_service)]


def create_app(service: AgentService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        configure_logging(settings.log_level, settings.log_format)
        if service is not None:  # injected (tests)
            app.state.service = service
            yield
            return
        with postgres_checkpointer(settings.psycopg_dsn) as checkpointer:
            app.state.service = build_service(checkpointer=checkpointer, settings=settings)
            logger.info("DealDesk API ready", extra={"model": app.state.service.spec.label})
            yield

    app = FastAPI(
        title="DealDesk",
        version="0.1.0",
        description="CRM assistant agent with human-approved writes.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------- dependencies -------

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")

    protected = [Depends(require_api_key)]

    # ------------------------------------------------------------------ routes --------

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(svc: Service) -> HealthResponse:
        db_ok = True
        try:
            with session_scope(svc.toolkit.session_factory) as session:
                session.execute(text("SELECT 1"))
        except Exception:
            logger.exception("health check: database unavailable")
            db_ok = False
        return HealthResponse(
            status="ok" if db_ok else "degraded", database=db_ok, model=svc.spec.label
        )

    @app.post("/v1/chat", response_model=TurnResponse, dependencies=protected, tags=["agent"])
    def chat(body: ChatRequest, svc: Service) -> TurnResponse:
        if body.thread_id and svc.pending_actions(body.thread_id):
            raise HTTPException(
                status.HTTP_409_CONFLICT, "Thread is waiting for an approval decision"
            )
        return _turn(svc.chat(body.message, body.thread_id, tags=["api"]))

    @app.post(
        "/v1/threads/{thread_id}/decision",
        response_model=TurnResponse,
        dependencies=protected,
        tags=["agent"],
    )
    def decide(thread_id: str, body: DecisionRequest, svc: Service) -> TurnResponse:
        try:
            result = svc.decide(thread_id, body.decision, body.reason, tags=["api"])
        except NoPendingApprovalError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return _turn(result)

    @app.get(
        "/v1/threads/{thread_id}",
        response_model=ThreadResponse,
        dependencies=protected,
        tags=["agent"],
    )
    def get_thread(thread_id: str, svc: Service) -> ThreadResponse:
        messages = svc.history(thread_id)
        pending = svc.pending_actions(thread_id)
        state = "awaiting_approval" if pending else ("completed" if messages else "empty")
        return ThreadResponse(
            thread_id=thread_id,
            status=state,
            pending_actions=[PendingAction(**a) for a in pending],
            messages=[
                MessageView(
                    role=m.type,
                    content=m.text,
                    tool_calls=[
                        {"name": c["name"], "args": c["args"]}
                        for c in (m.tool_calls if isinstance(m, AIMessage) else [])
                    ],
                )
                for m in messages
            ],
        )

    @app.post(
        "/v1/deals/{deal_id}/summary",
        response_model=DealSummary,
        dependencies=protected,
        tags=["crm"],
    )
    def summarize_deal(deal_id: int, svc: Service) -> DealSummary:
        try:
            return svc.toolkit.summarize_deal(deal_id)
        except NotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    @app.post(
        "/v1/webhooks/deal-stage-changed",
        response_model=WebhookAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["webhooks"],
    )
    async def deal_stage_changed(
        request: Request,
        background: BackgroundTasks,
        x_dealdesk_signature: Annotated[str | None, Header()] = None,
    ) -> WebhookAccepted:
        # Verify the HMAC over the raw body before trusting anything in it.
        raw = await request.body()
        if not verify_signature(
            raw, x_dealdesk_signature, settings.webhook_secret.get_secret_value()
        ):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid signature")
        event = DealStageChangedEvent.model_validate_json(raw)
        svc: AgentService = request.app.state.service
        thread_id = f"webhook-deal-{event.deal_id}-{event.event_id}"
        # Idempotency: webhook senders retry, so the same event must not run twice.
        if svc.history(thread_id):
            return WebhookAccepted(status="duplicate", thread_id=thread_id)
        background.add_task(_handle_stage_change, svc, event, thread_id)
        return WebhookAccepted(status="accepted", thread_id=thread_id)

    return app


def _turn(result: TurnResult) -> TurnResponse:
    return TurnResponse(
        thread_id=result.thread_id,
        status=result.status,
        answer=result.answer,
        pending_actions=[PendingAction(**a) for a in result.pending_actions],
        tool_calls=result.tool_calls,
    )


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    return bool(signature) and hmac.compare_digest(sign(body, secret), signature or "")


def _handle_stage_change(svc: AgentService, event: DealStageChangedEvent, thread_id: str) -> None:
    """Sync the stage into our data model, then let the agent propose next steps.

    Any writes the agent proposes pause for approval in this thread; a human reviews them
    via GET /v1/threads/{thread_id} and POST .../decision.
    """
    try:
        with session_scope(svc.toolkit.session_factory) as session:
            repo = CRMRepository(session)
            # Already in that stage -> nothing to do; the sync is idempotent.
            with contextlib.suppress(ValidationError):
                repo.update_deal_stage(event.deal_id, event.new_stage, source=Source.WEBHOOK)
        svc.chat(
            f"Deal #{event.deal_id} was just moved to stage '{event.new_stage.label}'. "
            "Review the deal and propose the most useful next step. If a follow-up task "
            "is needed, create it.",
            thread_id,
            tags=["webhook"],
            metadata={"event_id": event.event_id},
        )
    except NotFoundError:
        logger.warning("webhook for unknown deal", extra={"deal_id": event.deal_id})
    except Exception:
        logger.exception("webhook handling failed", extra={"thread_id": thread_id})
