"""LangChain tools over the CRM repository.

Design notes
- Each tool call runs in its own database transaction (unit of work), so a write is
  committed completely or not at all.
- Tools return compact JSON strings. Domain errors (unknown deal, past due date) come back
  as `{"error": ...}` so the model can correct itself instead of the run crashing.
- Read and write tools are kept in separate lists. The graph uses that split to require
  human approval for every write.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from dealdesk.agent.outputs import DealSummary, FollowUpEmail
from dealdesk.crm.repository import Clock, CRMRepository, NotFoundError, ValidationError, _utcnow
from dealdesk.db.models import ActivityType, DealStage, Source, TaskPriority, TaskStatus
from dealdesk.db.session import session_scope

# ----------------------------------------------------------------------------- inputs ---


class SearchCompaniesInput(BaseModel):
    query: str = Field(description="Part of the company name or domain, e.g. 'seeblick'.")


class CompanyIdInput(BaseModel):
    company_id: int = Field(description="Company ID from search_companies.")


class ListDealsInput(BaseModel):
    stage: DealStage | None = Field(default=None, description="Filter by pipeline stage ID.")
    owner: str | None = Field(default=None, description="Deal owner name (partial match).")
    company_id: int | None = Field(default=None, description="Only deals of this company.")
    open_only: bool = Field(default=False, description="Exclude closed won/lost deals.")
    stale_days: int | None = Field(
        default=None, description="Only open deals with no activity for at least N days."
    )
    min_amount: float | None = Field(default=None, description="Minimum deal amount in EUR.")
    closing_within_days: int | None = Field(
        default=None, description="Only open deals whose close date is within the next N days."
    )


class DealIdInput(BaseModel):
    deal_id: int = Field(description="Deal ID.")


class PipelineSummaryInput(BaseModel):
    owner: str | None = Field(default=None, description="Limit to one owner (partial name).")


class ListTasksInput(BaseModel):
    owner: str | None = Field(default=None, description="Task owner name (partial match).")
    deal_id: int | None = Field(default=None, description="Only tasks for this deal.")
    include_completed: bool = Field(default=False, description="Also return completed tasks.")
    overdue_only: bool = Field(default=False, description="Only open tasks past their due date.")


class DraftEmailInput(BaseModel):
    deal_id: int = Field(description="Deal ID to write the follow-up for.")
    goal: str = Field(description="What the email should achieve, e.g. 'book a pricing call'.")


class LogActivityInput(BaseModel):
    deal_id: int = Field(description="Deal ID.")
    type: ActivityType = Field(description="EMAIL, CALL, MEETING or NOTE.")
    summary: str = Field(description="What happened, in one or two sentences.")
    contact_id: int | None = Field(default=None, description="Contact involved, if any.")


class CreateTaskInput(BaseModel):
    deal_id: int = Field(description="Deal ID.")
    title: str = Field(description="Short, actionable task title.")
    due_date: date = Field(description="Due date in ISO format (YYYY-MM-DD), today or later.")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="LOW, MEDIUM, HIGH.")


class UpdateDealStageInput(BaseModel):
    deal_id: int = Field(description="Deal ID.")
    stage: DealStage = Field(description="New pipeline stage ID.")


# --------------------------------------------------------------------------- toolkit ---


def _to_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(exclude_none=True)
    if isinstance(obj, list):
        return json.dumps(
            [o.model_dump(mode="json", exclude_none=True) for o in obj], ensure_ascii=False
        )
    return json.dumps(obj, default=str, ensure_ascii=False)


@dataclass
class CRMToolkit:
    """Builds the agent's tools. Dependencies are injected so tests can swap them."""

    session_factory: sessionmaker[Session]
    model: BaseChatModel | None = None  # used by the drafting / summary tools
    clock: Clock = _utcnow
    write_source: Source = Source.AGENT

    # -- plumbing ----------------------------------------------------------------------
    def _run(self, fn: Callable[[CRMRepository], Any]) -> str:
        try:
            with session_scope(self.session_factory) as session:
                return _to_json(fn(CRMRepository(session, clock=self.clock)))
        except (NotFoundError, ValidationError) as exc:
            return json.dumps({"error": str(exc)})

    # -- reads -------------------------------------------------------------------------
    def search_companies(self, query: str) -> str:
        """Find companies by (partial) name or domain. Use this first when the user names
        a company, to get its ID."""
        return self._run(lambda r: r.search_companies(query))

    def get_company_overview(self, company_id: int) -> str:
        """Get a company's details, contacts (with contact IDs) and all its deals."""
        return self._run(lambda r: r.get_company_overview(company_id))

    def list_deals(
        self,
        stage: DealStage | None = None,
        owner: str | None = None,
        company_id: int | None = None,
        open_only: bool = False,
        stale_days: int | None = None,
        min_amount: float | None = None,
        closing_within_days: int | None = None,
    ) -> str:
        """List deals with optional filters. Results include stage, amount (EUR), close
        date, owner and days since the last activity. Sorted by amount, largest first."""
        return self._run(
            lambda r: r.list_deals(
                stage=stage,
                owner=owner,
                company_id=company_id,
                open_only=open_only,
                stale_days=stale_days,
                min_amount=min_amount,
                closing_within_days=closing_within_days,
            )
        )

    def get_deal(self, deal_id: int) -> str:
        """Get one deal with its full activity timeline, contacts and tasks."""
        return self._run(lambda r: r.get_deal(deal_id))

    def get_pipeline_summary(self, owner: str | None = None) -> str:
        """Pipeline totals: open deal count and value, won/lost value, and breakdowns by
        stage and by owner. Use this for any 'how much / how many' pipeline question."""
        return self._run(lambda r: r.pipeline_summary(owner=owner))

    def list_tasks(
        self,
        owner: str | None = None,
        deal_id: int | None = None,
        include_completed: bool = False,
        overdue_only: bool = False,
    ) -> str:
        """List tasks (open by default) with due dates, priority and an is_overdue flag."""
        status = None if include_completed else TaskStatus.NOT_STARTED
        return self._run(
            lambda r: r.list_tasks(
                owner=owner, deal_id=deal_id, status=status, overdue_only=overdue_only
            )
        )

    def draft_follow_up_email(self, deal_id: int, goal: str) -> str:
        """Draft (not send) a follow-up email for a deal, grounded in its CRM history."""
        if self.model is None:
            return json.dumps({"error": "Drafting is not available: no model configured."})
        with session_scope(self.session_factory) as session:
            try:
                detail = CRMRepository(session, clock=self.clock).get_deal(deal_id)
            except NotFoundError as exc:
                return json.dumps({"error": str(exc)})
        drafter = self.model.with_structured_output(FollowUpEmail)
        email = drafter.invoke(
            "Write a concise, friendly B2B follow-up email in English for this deal. "
            "Use only facts from the CRM record; do not invent prices or dates. "
            f"Goal: {goal}\n\nCRM record (JSON):\n{detail.model_dump_json()}",
            config={"run_name": "draft_follow_up_email"},
        )
        return _to_json(email)

    def summarize_deal(self, deal_id: int) -> DealSummary:
        """Structured deal brief (used by the REST API, not exposed as an agent tool)."""
        if self.model is None:
            raise RuntimeError("No model configured")
        with session_scope(self.session_factory) as session:
            detail = CRMRepository(session, clock=self.clock).get_deal(deal_id)
        summarizer = self.model.with_structured_output(DealSummary)
        result = summarizer.invoke(
            f"Today is {self.clock().date().isoformat()}. Summarise this CRM deal for a sales "
            "manager. Use only the data given.\n\n"
            f"{detail.model_dump_json()}",
            config={"run_name": "summarize_deal"},
        )
        assert isinstance(result, DealSummary)
        return result

    # -- writes (always behind human approval in the graph) ------------------------------
    def log_activity(
        self, deal_id: int, type: ActivityType, summary: str, contact_id: int | None = None
    ) -> str:
        """Log an email, call, meeting or note on a deal's timeline. Requires approval."""
        return self._run(
            lambda r: r.log_activity(
                deal_id, type, summary, contact_id=contact_id, source=self.write_source
            )
        )

    def create_task(
        self,
        deal_id: int,
        title: str,
        due_date: date,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> str:
        """Create a follow-up task on a deal, assigned to the deal owner. Requires approval."""
        return self._run(
            lambda r: r.create_task(
                deal_id, title, due_date, priority=priority, source=self.write_source
            )
        )

    def update_deal_stage(self, deal_id: int, stage: DealStage) -> str:
        """Move a deal to another pipeline stage. Requires approval."""
        return self._run(lambda r: r.update_deal_stage(deal_id, stage, source=self.write_source))

    # -- assembly ----------------------------------------------------------------------
    def read_tools(self) -> list[BaseTool]:
        specs: list[tuple[Callable[..., str], type[BaseModel]]] = [
            (self.search_companies, SearchCompaniesInput),
            (self.get_company_overview, CompanyIdInput),
            (self.list_deals, ListDealsInput),
            (self.get_deal, DealIdInput),
            (self.get_pipeline_summary, PipelineSummaryInput),
            (self.list_tasks, ListTasksInput),
            (self.draft_follow_up_email, DraftEmailInput),
        ]
        return [_make_tool(fn, schema) for fn, schema in specs]

    def write_tools(self) -> list[BaseTool]:
        specs: list[tuple[Callable[..., str], type[BaseModel]]] = [
            (self.log_activity, LogActivityInput),
            (self.create_task, CreateTaskInput),
            (self.update_deal_stage, UpdateDealStageInput),
        ]
        return [_make_tool(fn, schema) for fn, schema in specs]

    def all_tools(self) -> list[BaseTool]:
        return self.read_tools() + self.write_tools()


WRITE_TOOL_NAMES = frozenset({"log_activity", "create_task", "update_deal_stage"})


def _make_tool(fn: Callable[..., str], schema: type[BaseModel]) -> BaseTool:
    return StructuredTool.from_function(
        func=fn,
        name=fn.__name__,
        description=(fn.__doc__ or "").strip(),
        args_schema=schema,
    )


def describe_action(tool_name: str, args: dict[str, Any]) -> str:
    """Human-readable description of a proposed write, shown in the approval prompt."""
    deal = f"deal #{args.get('deal_id')}"
    if tool_name == "create_task":
        return (
            f"Create task '{args.get('title')}' on {deal}, due {args.get('due_date')} "
            f"(priority {args.get('priority', 'MEDIUM')})"
        )
    if tool_name == "log_activity":
        return f"Log {args.get('type')} on {deal}: {args.get('summary')}"
    if tool_name == "update_deal_stage":
        stage = args.get("stage")
        label = DealStage(str(stage)).label if stage in DealStage._value2member_map_ else stage
        return f"Move {deal} to stage '{label}'"
    return f"{tool_name}({args})"
