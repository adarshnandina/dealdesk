"""CRM data access.

The repository is the only place that knows about SQL. Tools, the API and the MCP server
all go through it, so swapping the mock Postgres CRM for a real HubSpot client means
writing one new class with the same methods.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Subquery, func, or_, select
from sqlalchemy.orm import Session, joinedload

from dealdesk.crm.schemas import (
    ActivityOut,
    CompanyOverview,
    CompanySummary,
    ContactOut,
    DealDetail,
    DealRow,
    OwnerTotal,
    PipelineSummary,
    StageTotal,
    TaskOut,
)
from dealdesk.db.models import (
    Activity,
    ActivityType,
    Company,
    Contact,
    Deal,
    DealStage,
    Owner,
    Source,
    Task,
    TaskPriority,
    TaskStatus,
)

Clock = Callable[[], datetime]
OPEN_STAGES = [s for s in DealStage if not s.is_closed]


class NotFoundError(LookupError):
    """Raised when a referenced CRM record does not exist."""


class ValidationError(ValueError):
    """Raised when a write request is not allowed by business rules."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CRMRepository:
    def __init__(self, session: Session, clock: Clock = _utcnow) -> None:
        self.session = session
        self.clock = clock

    # ------------------------------------------------------------------ reads ----------

    def search_companies(self, query: str, limit: int = 10) -> list[CompanySummary]:
        pattern = f"%{query.strip()}%"
        stmt = (
            select(Company)
            .options(joinedload(Company.owner))
            .where(or_(Company.name.ilike(pattern), Company.domain.ilike(pattern)))
            .order_by(Company.name)
            .limit(limit)
        )
        return [self._company(c) for c in self.session.scalars(stmt)]

    def get_company_overview(self, company_id: int) -> CompanyOverview:
        company = self.session.get(Company, company_id, options=[joinedload(Company.owner)])
        if company is None:
            raise NotFoundError(f"Company {company_id} not found")
        contacts = self.session.scalars(
            select(Contact).where(Contact.company_id == company_id).order_by(Contact.id)
        )
        return CompanyOverview(
            company=self._company(company),
            contacts=[self._contact(c, company.name) for c in contacts],
            deals=self.list_deals(company_id=company_id),
        )

    def list_deals(
        self,
        *,
        stage: DealStage | None = None,
        owner: str | None = None,
        company_id: int | None = None,
        open_only: bool = False,
        stale_days: int | None = None,
        min_amount: float | None = None,
        closing_within_days: int | None = None,
        limit: int = 50,
    ) -> list[DealRow]:
        last_activity = self._last_activity_subquery()
        stmt = (
            select(Deal, last_activity.c.last_at)
            .outerjoin(last_activity, last_activity.c.deal_id == Deal.id)
            .options(joinedload(Deal.company), joinedload(Deal.owner))
        )
        if stage is not None:
            stmt = stmt.where(Deal.stage == stage)
        if open_only or stale_days is not None or closing_within_days is not None:
            stmt = stmt.where(Deal.stage.in_(OPEN_STAGES))
        if owner:
            stmt = stmt.join(Owner, Deal.owner_id == Owner.id).where(
                Owner.name.ilike(f"%{owner.strip()}%")
            )
        if company_id is not None:
            stmt = stmt.where(Deal.company_id == company_id)
        if min_amount is not None:
            stmt = stmt.where(Deal.amount >= Decimal(str(min_amount)))
        if stale_days is not None:
            cutoff = self.clock() - timedelta(days=stale_days)
            stmt = stmt.where(
                or_(last_activity.c.last_at.is_(None), last_activity.c.last_at < cutoff)
            )
        if closing_within_days is not None:
            today = self.clock().date()
            stmt = stmt.where(
                Deal.close_date.between(today, today + timedelta(closing_within_days))
            )
        stmt = stmt.order_by(Deal.amount.desc(), Deal.id).limit(limit)
        return [self._deal_row(deal, last_at) for deal, last_at in self.session.execute(stmt)]

    def get_deal(self, deal_id: int) -> DealDetail:
        deal = self._load_deal(deal_id)
        activities = [self._activity(a) for a in deal.activities]
        last_at = max((a.occurred_at for a in activities), default=None)
        contacts = self.session.scalars(
            select(Contact).where(Contact.company_id == deal.company_id).order_by(Contact.id)
        )
        return DealDetail(
            deal=self._deal_row(deal, last_at),
            contacts=[self._contact(c, deal.company.name) for c in contacts],
            activities=activities,
            tasks=[self._task(t) for t in deal.tasks],
        )

    def pipeline_summary(self, owner: str | None = None) -> PipelineSummary:
        deals = self.list_deals(owner=owner, limit=10_000)
        by_stage: dict[DealStage, list[DealRow]] = defaultdict(list)
        by_owner: dict[str, list[DealRow]] = defaultdict(list)
        for d in deals:
            by_stage[d.stage].append(d)
            if not d.is_closed:
                by_owner[d.owner or "Unassigned"].append(d)
        open_deals = [d for d in deals if not d.is_closed]
        return PipelineSummary(
            currency="EUR",
            open_count=len(open_deals),
            open_amount=sum(d.amount for d in open_deals),
            won_amount=sum(d.amount for d in by_stage[DealStage.CLOSED_WON]),
            lost_amount=sum(d.amount for d in by_stage[DealStage.CLOSED_LOST]),
            by_stage=[
                StageTotal(
                    stage=s,
                    label=s.label,
                    count=len(by_stage[s]),
                    amount=sum(d.amount for d in by_stage[s]),
                )
                for s in DealStage
            ],
            by_owner=[
                OwnerTotal(owner=o, open_count=len(ds), open_amount=sum(d.amount for d in ds))
                for o, ds in sorted(by_owner.items())
            ],
        )

    def list_tasks(
        self,
        *,
        owner: str | None = None,
        deal_id: int | None = None,
        status: TaskStatus | None = TaskStatus.NOT_STARTED,
        overdue_only: bool = False,
        limit: int = 50,
    ) -> list[TaskOut]:
        stmt = select(Task).options(joinedload(Task.deal), joinedload(Task.owner))
        if status is not None:
            stmt = stmt.where(Task.status == status)
        if deal_id is not None:
            stmt = stmt.where(Task.deal_id == deal_id)
        if owner:
            stmt = stmt.join(Owner, Task.owner_id == Owner.id).where(
                Owner.name.ilike(f"%{owner.strip()}%")
            )
        if overdue_only:
            stmt = stmt.where(
                Task.status == TaskStatus.NOT_STARTED, Task.due_date < self.clock().date()
            )
        stmt = stmt.order_by(Task.due_date, Task.id).limit(limit)
        return [self._task(t) for t in self.session.scalars(stmt)]

    # ----------------------------------------------------------------- writes ----------

    def log_activity(
        self,
        deal_id: int,
        type: ActivityType,
        summary: str,
        *,
        contact_id: int | None = None,
        source: Source = Source.AGENT,
    ) -> ActivityOut:
        deal = self._load_deal(deal_id)
        if not summary.strip():
            raise ValidationError("Activity summary must not be empty")
        if contact_id is not None:
            contact = self.session.get(Contact, contact_id)
            if contact is None or contact.company_id != deal.company_id:
                raise ValidationError(f"Contact {contact_id} does not belong to deal {deal_id}")
        activity = Activity(
            deal_id=deal.id,
            contact_id=contact_id,
            type=type,
            summary=summary.strip(),
            occurred_at=self.clock(),
            source=source,
        )
        self.session.add(activity)
        self.session.flush()
        self.session.refresh(activity)
        return self._activity(activity)

    def create_task(
        self,
        deal_id: int,
        title: str,
        due_date: date,
        *,
        priority: TaskPriority = TaskPriority.MEDIUM,
        source: Source = Source.AGENT,
    ) -> TaskOut:
        deal = self._load_deal(deal_id)
        if not title.strip():
            raise ValidationError("Task title must not be empty")
        if due_date < self.clock().date():
            raise ValidationError(f"Due date {due_date} is in the past")
        task = Task(
            deal_id=deal.id,
            owner_id=deal.owner_id,  # tasks default to the deal owner, as in HubSpot
            title=title.strip(),
            due_date=due_date,
            priority=priority,
            source=source,
        )
        self.session.add(task)
        self.session.flush()
        self.session.refresh(task)
        return self._task(task)

    def update_deal_stage(
        self, deal_id: int, stage: DealStage, *, source: Source = Source.AGENT
    ) -> DealRow:
        deal = self._load_deal(deal_id)
        previous = deal.stage
        if previous == stage:
            raise ValidationError(f"Deal {deal_id} is already in stage '{stage.label}'")
        deal.stage = stage
        # Keep an audit trail on the deal timeline, like HubSpot's property history.
        self.session.add(
            Activity(
                deal_id=deal.id,
                type=ActivityType.NOTE,
                summary=f"Stage changed from '{previous.label}' to '{stage.label}' ({source}).",
                occurred_at=self.clock(),
                source=source,
            )
        )
        self.session.flush()
        return self.get_deal(deal_id).deal

    # ---------------------------------------------------------------- helpers ----------

    def _load_deal(self, deal_id: int) -> Deal:
        deal = self.session.get(
            Deal,
            deal_id,
            options=[joinedload(Deal.company), joinedload(Deal.owner)],
            populate_existing=True,
        )
        if deal is None:
            raise NotFoundError(f"Deal {deal_id} not found")
        return deal

    @staticmethod
    def _last_activity_subquery() -> Subquery:
        return (
            select(Activity.deal_id, func.max(Activity.occurred_at).label("last_at"))
            .group_by(Activity.deal_id)
            .subquery()
        )

    def _deal_row(self, deal: Deal, last_at: datetime | None) -> DealRow:
        days = (self.clock() - last_at).days if last_at else None
        return DealRow(
            id=deal.id,
            name=deal.name,
            company_id=deal.company_id,
            company=deal.company.name,
            owner=deal.owner.name if deal.owner else None,
            stage=deal.stage,
            stage_label=deal.stage.label,
            is_closed=deal.stage.is_closed,
            amount=float(deal.amount),
            currency=deal.currency,
            close_date=deal.close_date,
            last_activity_at=last_at,
            days_since_last_activity=days,
        )

    @staticmethod
    def _company(c: Company) -> CompanySummary:
        return CompanySummary(
            id=c.id,
            name=c.name,
            domain=c.domain,
            industry=c.industry,
            city=c.city,
            country=c.country,
            owner=c.owner.name if c.owner else None,
        )

    @staticmethod
    def _contact(c: Contact, company: str) -> ContactOut:
        return ContactOut(
            id=c.id, name=c.full_name, email=c.email, job_title=c.job_title, company=company
        )

    @staticmethod
    def _activity(a: Activity) -> ActivityOut:
        return ActivityOut(
            id=a.id,
            deal_id=a.deal_id,
            type=a.type,
            summary=a.summary,
            occurred_at=a.occurred_at,
            contact=a.contact.full_name if a.contact else None,
            source=a.source,
        )

    def _task(self, t: Task) -> TaskOut:
        return TaskOut(
            id=t.id,
            deal_id=t.deal_id,
            deal=t.deal.name,
            title=t.title,
            due_date=t.due_date,
            status=t.status,
            priority=t.priority,
            owner=t.owner.name if t.owner else None,
            is_overdue=t.status == TaskStatus.NOT_STARTED and t.due_date < self.clock().date(),
            source=t.source,
        )
