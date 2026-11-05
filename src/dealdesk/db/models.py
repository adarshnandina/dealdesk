"""SQLAlchemy ORM models for a HubSpot-shaped CRM.

Object names and pipeline stage IDs follow HubSpot's defaults (owners, companies,
contacts, deals, engagements/activities, tasks), so a real HubSpot adapter could later
replace the mock database without changing the agent's tools.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class DealStage(StrEnum):
    """HubSpot's default sales pipeline, using its internal stage IDs."""

    APPOINTMENT_SCHEDULED = "appointmentscheduled"
    QUALIFIED_TO_BUY = "qualifiedtobuy"
    PRESENTATION_SCHEDULED = "presentationscheduled"
    DECISION_MAKER_BOUGHT_IN = "decisionmakerboughtin"
    CONTRACT_SENT = "contractsent"
    CLOSED_WON = "closedwon"
    CLOSED_LOST = "closedlost"

    @property
    def label(self) -> str:
        return _STAGE_LABELS[self]

    @property
    def is_closed(self) -> bool:
        return self in (DealStage.CLOSED_WON, DealStage.CLOSED_LOST)


_STAGE_LABELS = {
    DealStage.APPOINTMENT_SCHEDULED: "Appointment scheduled",
    DealStage.QUALIFIED_TO_BUY: "Qualified to buy",
    DealStage.PRESENTATION_SCHEDULED: "Presentation scheduled",
    DealStage.DECISION_MAKER_BOUGHT_IN: "Decision maker bought-in",
    DealStage.CONTRACT_SENT: "Contract sent",
    DealStage.CLOSED_WON: "Closed won",
    DealStage.CLOSED_LOST: "Closed lost",
}


class ActivityType(StrEnum):
    EMAIL = "EMAIL"
    CALL = "CALL"
    MEETING = "MEETING"
    NOTE = "NOTE"


class TaskStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    COMPLETED = "COMPLETED"


class TaskPriority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Source(StrEnum):
    """Who created a record. Lets us audit everything the agent wrote."""

    SEED = "seed"
    USER = "user"
    AGENT = "agent"
    WEBHOOK = "webhook"


def _enum(enum_cls: type[StrEnum], name: str) -> Enum:
    # VARCHAR + CHECK constraint instead of a native Postgres ENUM: adding a value later
    # is a simple migration, and the schema stays portable.
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Owner(Base, TimestampMixin):
    __tablename__ = "owners"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True)


class Company(Base, TimestampMixin):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    domain: Mapped[str | None] = mapped_column(String(200), unique=True)
    industry: Mapped[str | None] = mapped_column(String(100))
    city: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(2))
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("owners.id"))

    owner: Mapped[Owner | None] = relationship()
    contacts: Mapped[list[Contact]] = relationship(back_populates="company")
    deals: Mapped[list[Deal]] = relationship(back_populates="company")


class Contact(Base, TimestampMixin):
    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    job_title: Mapped[str | None] = mapped_column(String(150))

    company: Mapped[Company] = relationship(back_populates="contacts")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class Deal(Base, TimestampMixin):
    __tablename__ = "deals"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("owners.id"))
    stage: Mapped[DealStage] = mapped_column(_enum(DealStage, "deal_stage"), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3), default="EUR")
    close_date: Mapped[date | None] = mapped_column(Date)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    company: Mapped[Company] = relationship(back_populates="deals")
    owner: Mapped[Owner | None] = relationship()
    activities: Mapped[list[Activity]] = relationship(
        back_populates="deal", order_by="Activity.occurred_at.desc()"
    )
    tasks: Mapped[list[Task]] = relationship(back_populates="deal", order_by="Task.due_date")


class Activity(Base, TimestampMixin):
    """A logged engagement (email, call, meeting, note) on a deal."""

    __tablename__ = "activities"
    __table_args__ = (Index("ix_activities_deal_occurred", "deal_id", "occurred_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    deal_id: Mapped[int] = mapped_column(ForeignKey("deals.id", ondelete="CASCADE"))
    contact_id: Mapped[int | None] = mapped_column(ForeignKey("contacts.id", ondelete="SET NULL"))
    type: Mapped[ActivityType] = mapped_column(_enum(ActivityType, "activity_type"))
    summary: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[Source] = mapped_column(_enum(Source, "record_source"), default=Source.USER)

    deal: Mapped[Deal] = relationship(back_populates="activities")
    contact: Mapped[Contact | None] = relationship()


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    deal_id: Mapped[int] = mapped_column(ForeignKey("deals.id", ondelete="CASCADE"))
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("owners.id"))
    title: Mapped[str] = mapped_column(String(300))
    due_date: Mapped[date] = mapped_column(Date)
    status: Mapped[TaskStatus] = mapped_column(
        _enum(TaskStatus, "task_status"), default=TaskStatus.NOT_STARTED
    )
    priority: Mapped[TaskPriority] = mapped_column(
        _enum(TaskPriority, "task_priority"), default=TaskPriority.MEDIUM
    )
    source: Mapped[Source] = mapped_column(_enum(Source, "record_source"), default=Source.USER)

    deal: Mapped[Deal] = relationship(back_populates="tasks")
    owner: Mapped[Owner | None] = relationship()
