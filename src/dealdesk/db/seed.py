"""Deterministic, hand-written demo data for a small DACH/Nordics B2B sales team.

The data is hand-written rather than random so that evaluation questions have fixed,
checkable answers (for example "which open deals have had no activity for 21+ days?" is
always Fjordway, Lindqvist and Ostsee). Dates are relative to the seeding time, so the
data never looks out of date. All companies and people are fictional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.orm import Session

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

OWNERS = [
    ("Lena Fischer", "lena.fischer@dealdesk.example"),
    ("Jonas Berg", "jonas.berg@dealdesk.example"),
    ("Mia Schneider", "mia.schneider@dealdesk.example"),
]

# key: (name, domain, industry, city, country, owner index)
COMPANIES: dict[str, tuple[str, str, str, str, str, int]] = {
    "nordlicht": ("Nordlicht Energie GmbH", "nordlicht-energie.example", "Energy", "Hamburg", "DE", 0),
    "alpenblick": ("Alpenblick Hotels AG", "alpenblick-hotels.example", "Hospitality", "Innsbruck", "AT", 2),
    "fjordway": ("Fjordway Logistics AS", "fjordway.example", "Logistics", "Bergen", "NO", 1),
    "kranich": ("Kranich Medtech GmbH", "kranich-medtech.example", "Healthcare", "Munich", "DE", 0),
    "seeblick": ("Seeblick Versicherung AG", "seeblick-versicherung.example", "Insurance", "Zurich", "CH", 0),
    "birkenwald": ("Birkenwald Software GmbH", "birkenwald.example", "Software", "Stuttgart", "DE", 2),
    "lindqvist": ("Lindqvist Retail AB", "lindqvist-retail.example", "Retail", "Gothenburg", "SE", 1),
    "rheinkraft": ("Rheinkraft Maschinenbau GmbH", "rheinkraft.example", "Manufacturing", "Cologne", "DE", 2),
    "bergquell": ("Bergquell Foods AG", "bergquell-foods.example", "Food & Beverage", "Bern", "CH", 0),
    "ostsee": ("Ostsee Mobility GmbH", "ostsee-mobility.example", "Mobility", "Rostock", "DE", 1),
    "sundby": ("Sundby Fintech ApS", "sundby.example", "Financial Services", "Copenhagen", "DK", 1),
    "tannenhof": ("Tannenhof Bildung gGmbH", "tannenhof-bildung.example", "Education", "Freiburg", "DE", 2),
}  # fmt: skip

# company key -> [(first, last, job title)]
CONTACTS: dict[str, list[tuple[str, str, str]]] = {
    "nordlicht": [("Katrin", "Vogel", "Head of Customer Service"), ("Paul", "Hartmann", "CIO")],
    "alpenblick": [("Sophie", "Gruber", "Director of Sales")],
    "fjordway": [("Erik", "Solberg", "COO"), ("Ingrid", "Dahl", "IT Project Lead")],
    "kranich": [("Thomas", "Weber", "Head of Digital")],
    "seeblick": [("Andrea", "Keller", "Head of Claims"), ("Marco", "Brunner", "CTO")],
    "birkenwald": [("Julia", "Roth", "VP Partnerships")],
    "lindqvist": [("Anna", "Lindqvist", "CMO")],
    "rheinkraft": [("Stefan", "Koch", "Head of After-Sales")],
    "bergquell": [("Claudia", "Meier", "Head of Marketing")],
    "ostsee": [("Felix", "Neumann", "Head of Operations")],
    "sundby": [("Mads", "Sorensen", "CEO")],
    "tannenhof": [("Laura", "Becker", "Managing Director")],
}


@dataclass(frozen=True)
class SeedActivity:
    days_ago: int
    type: ActivityType
    summary: str
    contact: int | None = 0  # index into the company's contacts


@dataclass(frozen=True)
class SeedTask:
    title: str
    due_in_days: int
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.NOT_STARTED


@dataclass(frozen=True)
class SeedDeal:
    name: str
    company: str
    owner: int
    stage: DealStage
    amount: int
    close_in_days: int
    activities: list[SeedActivity] = field(default_factory=list)
    tasks: list[SeedTask] = field(default_factory=list)


A, T = SeedActivity, SeedTask
S = DealStage
E, C, M, N = ActivityType.EMAIL, ActivityType.CALL, ActivityType.MEETING, ActivityType.NOTE
HIGH = TaskPriority.HIGH

# Deal IDs follow list order (1-based) after a reset.
DEALS: list[SeedDeal] = [
    SeedDeal("Service hub AI chatbot", "nordlicht", 0, S.PRESENTATION_SCHEDULED, 85_000, 40, [
        A(30, N, "Kickoff: Nordlicht wants an AI chatbot for its customer service hub.", None),
        A(12, M, "Discovery workshop with Katrin. Main pain point: 40% of tickets are tariff questions."),
        A(4, E, "Sent Katrin the agenda for the solution presentation."),
    ], [T("Prepare chatbot demo for Katrin", 3, HIGH)]),
    SeedDeal("HubSpot data migration", "nordlicht", 0, S.CLOSED_WON, 42_000, -20, [
        A(60, C, "Scoping call with Paul about migrating from the legacy CRM.", 1),
        A(25, E, "Paul returned the signed contract.", 1),
    ]),
    SeedDeal("Booking assistant pilot", "alpenblick", 2, S.QUALIFIED_TO_BUY, 36_000, 60, [
        A(18, C, "Intro call with Sophie. Interested in a booking assistant for 3 hotels."),
        A(6, E, "Sent Sophie a hospitality case study."),
    ]),
    SeedDeal("Fleet CRM rollout", "fjordway", 1, S.DECISION_MAKER_BOUGHT_IN, 120_000, 25, [
        A(70, M, "On-site meeting with Erik and Ingrid in Bergen."),
        A(50, C, "Erik confirmed the budget for the fleet CRM rollout."),
        A(35, E, "Ingrid sent the IT security questionnaire; waiting on our answers.", 1),
    ], [T("Follow up on IT security questionnaire", -10, HIGH)]),
    SeedDeal("Patient onboarding portal", "kranich", 0, S.CLOSED_WON, 64_000, -9, [
        A(40, M, "Solution presentation for Thomas and the digital team."),
        A(15, E, "Final pricing agreed with Thomas."),
        A(9, N, "Contract signed. Handover to delivery team."),
    ], [T("Send onboarding kickoff invite", -8, status=TaskStatus.COMPLETED)]),
    SeedDeal("Clinic CRM expansion", "kranich", 0, S.APPOINTMENT_SCHEDULED, 28_000, 75, [
        A(3, E, "Thomas asked for a call about expanding the CRM to 4 more clinics."),
    ], [T("Schedule expansion call with Thomas", 2)]),
    SeedDeal("Claims intake agent", "seeblick", 0, S.CONTRACT_SENT, 180_000, 10, [
        A(55, M, "Workshop with Andrea on automating first notice of loss intake."),
        A(30, M, "Security review with Marco passed.", 1),
        A(8, E, "Sent the contract to Andrea and Marco."),
        A(2, C, "Marco's legal team has questions on the data processing agreement (DPA).", 1),
    ], [T("Answer DPA questions from Seeblick legal", 1, HIGH)]),
    SeedDeal("Partner portal migration", "birkenwald", 2, S.CONTRACT_SENT, 48_000, 14, [
        A(20, M, "Scoping meeting with Julia for the partner portal migration."),
        A(5, E, "Julia is reviewing the contract with procurement."),
    ]),
    SeedDeal("Loyalty agent pilot", "lindqvist", 1, S.PRESENTATION_SCHEDULED, 55_000, 30, [
        A(45, C, "Anna wants an AI agent for loyalty programme questions."),
        A(28, E, "Anna asked to postpone the presentation until after the Q3 campaign."),
    ]),
    SeedDeal("After-sales ticket automation", "rheinkraft", 2, S.DECISION_MAKER_BOUGHT_IN, 95_000, 20, [
        A(26, M, "Demo for Stefan's after-sales team. Strong interest."),
        A(11, C, "Stefan got sign-off from the managing director."),
        A(1, E, "Stefan requested final pricing for a 3-year term."),
    ], [T("Send 3-year pricing to Stefan", 2, HIGH)]),
    SeedDeal("Marketing hub onboarding", "bergquell", 0, S.QUALIFIED_TO_BUY, 22_000, 45, [
        A(9, C, "Claudia wants help onboarding the marketing team to HubSpot."),
        A(7, N, "Claudia's team of 6 currently uses spreadsheets for campaigns."),
    ]),
    SeedDeal("Service desk automation", "ostsee", 1, S.QUALIFIED_TO_BUY, 70_000, 15, [
        A(80, M, "First meeting with Felix about automating the service desk."),
        A(45, E, "Felix: the budget decision is expected next month."),
    ]),
    SeedDeal("KYC document agent", "sundby", 1, S.APPOINTMENT_SCHEDULED, 40_000, 90, [
        A(5, E, "Mads booked an intro meeting about a KYC document agent."),
    ]),
    SeedDeal("Student services CRM", "tannenhof", 2, S.CLOSED_LOST, 18_000, -30, [
        A(50, C, "Laura is comparing three vendors."),
        A(32, N, "Lost: chose a cheaper competitor because of budget constraints."),
    ]),
    SeedDeal("Revenue dashboard", "alpenblick", 2, S.CLOSED_LOST, 15_000, -45, [
        A(70, C, "Sophie asked for a revenue dashboard quote."),
        A(48, N, "Lost: project deprioritised by Alpenblick."),
    ]),
    SeedDeal("Broker portal chatbot", "seeblick", 0, S.APPOINTMENT_SCHEDULED, 60_000, 100, [
        A(14, E, "Andrea introduced us to the broker team for a portal chatbot."),
    ]),
    SeedDeal("Driver app support bot", "fjordway", 1, S.CLOSED_WON, 30_000, -60, [
        A(65, E, "Erik signed the order for the driver app support bot."),
    ]),
    SeedDeal("Sales agent workshop", "birkenwald", 2, S.PRESENTATION_SCHEDULED, 12_000, 20, [
        A(10, C, "Julia wants a workshop on AI agents for her sales team."),
        A(2, M, "Workshop agenda agreed with Julia."),
    ]),
]  # fmt: skip

TABLES = ["tasks", "activities", "deals", "contacts", "companies", "owners"]


def reset(session: Session) -> None:
    session.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))


def seed(session: Session, *, now: datetime | None = None, do_reset: bool = True) -> dict[str, int]:
    """Insert the demo dataset. Returns row counts per table."""
    now = now or datetime.now(UTC)
    today = now.date()
    if do_reset:
        reset(session)

    owners = [Owner(name=n, email=e) for n, e in OWNERS]
    session.add_all(owners)

    companies: dict[str, Company] = {}
    contacts: dict[str, list[Contact]] = {}
    for key, (name, domain, industry, city, country, owner_idx) in COMPANIES.items():
        company = Company(
            name=name,
            domain=domain,
            industry=industry,
            city=city,
            country=country,
            owner=owners[owner_idx],
        )
        companies[key] = company
        contacts[key] = [
            Contact(
                company=company,
                first_name=first,
                last_name=last,
                email=f"{first.lower()}.{last.lower()}@{domain}",
                job_title=title,
            )
            for first, last, title in CONTACTS[key]
        ]
    session.add_all(companies.values())
    session.flush()

    counts = {"owners": len(owners), "companies": len(companies), "deals": 0}
    counts["contacts"] = sum(len(c) for c in contacts.values())
    counts["activities"] = counts["tasks"] = 0
    for spec in DEALS:
        deal = Deal(
            name=spec.name,
            company=companies[spec.company],
            owner=owners[spec.owner],
            stage=spec.stage,
            amount=Decimal(spec.amount),
            currency="EUR",
            close_date=today + timedelta(days=spec.close_in_days),
        )
        session.add(deal)
        session.flush()  # assign IDs in list order
        for act in spec.activities:
            occurred = datetime.combine(today - timedelta(days=act.days_ago), time(10, 0), UTC)
            session.add(
                Activity(
                    deal=deal,
                    contact=None if act.contact is None else contacts[spec.company][act.contact],
                    type=act.type,
                    summary=act.summary,
                    occurred_at=occurred,
                    source=Source.SEED,
                )
            )
        for task in spec.tasks:
            session.add(
                Task(
                    deal=deal,
                    owner=owners[spec.owner],
                    title=task.title,
                    due_date=today + timedelta(days=task.due_in_days),
                    status=task.status,
                    priority=task.priority,
                    source=Source.SEED,
                )
            )
        counts["deals"] += 1
        counts["activities"] += len(spec.activities)
        counts["tasks"] += len(spec.tasks)
    session.flush()
    return counts
