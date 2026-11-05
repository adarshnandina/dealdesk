"""Pydantic DTOs returned by the repository.

Tools return these (as JSON) instead of ORM objects, so the LLM sees a stable, compact
shape and we never leak lazy-loaded database sessions across layers.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from dealdesk.db.models import ActivityType, DealStage, Source, TaskPriority, TaskStatus


class DTO(BaseModel):
    model_config = ConfigDict(frozen=True)


class CompanySummary(DTO):
    id: int
    name: str
    domain: str | None
    industry: str | None
    city: str | None
    country: str | None
    owner: str | None


class ContactOut(DTO):
    id: int
    name: str
    email: str
    job_title: str | None
    company: str


class DealRow(DTO):
    id: int
    name: str
    company_id: int
    company: str
    owner: str | None
    stage: DealStage
    stage_label: str
    is_closed: bool
    amount: float
    currency: str
    close_date: date | None
    last_activity_at: datetime | None
    days_since_last_activity: int | None


class ActivityOut(DTO):
    id: int
    deal_id: int
    type: ActivityType
    summary: str
    occurred_at: datetime
    contact: str | None
    source: Source


class TaskOut(DTO):
    id: int
    deal_id: int
    deal: str
    title: str
    due_date: date
    status: TaskStatus
    priority: TaskPriority
    owner: str | None
    is_overdue: bool
    source: Source


class DealDetail(DTO):
    deal: DealRow
    contacts: list[ContactOut]
    activities: list[ActivityOut]
    tasks: list[TaskOut]


class CompanyOverview(DTO):
    company: CompanySummary
    contacts: list[ContactOut]
    deals: list[DealRow]


class StageTotal(DTO):
    stage: DealStage
    label: str
    count: int
    amount: float


class OwnerTotal(DTO):
    owner: str
    open_count: int
    open_amount: float


class PipelineSummary(DTO):
    currency: str
    open_count: int
    open_amount: float
    won_amount: float
    lost_amount: float
    by_stage: list[StageTotal]
    by_owner: list[OwnerTotal]
