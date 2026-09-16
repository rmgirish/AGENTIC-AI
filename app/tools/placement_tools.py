import operator
from datetime import datetime, timezone

from app.domain import AlreadyApplied, Rule, Student
from app.data import InMemoryPlacementRepo
from app.tools.dispatch import dispatch

OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "in": lambda actual, allowed: actual in allowed.split(","),
}


def _passes(rule: Rule, student_value) -> bool:
    return OPS[rule.op](student_value, rule.typed_value())


def _unknown_student(roll_no: str) -> dict:
    return {"error": "unknown_student",
            "hint": f"No student with roll number {roll_no!r}. Ask the user for their roll number, e.g. 22CS045."}


def _unknown_drive(drive_id: int) -> dict:
    return {"error": "unknown_drive",
            "hint": f"No drive with id {drive_id}. Call list_open_drives to get valid ids."}


class PlacementTools:
    """Every method named in TOOL_NAMES is exposed to the model. Its docstring IS the prompt.

    Two tools are complete samples: check_eligibility (read-only) and apply_to_drive (side effect).
    Copy their patterns for the tools marked TODO.
    """

    READ_ONLY = ("list_open_drives", "get_student", "check_eligibility", "list_my_applications")
    SIDE_EFFECTS = ("apply_to_drive", "book_interview_slot", "notify_student")
    TOOL_NAMES = READ_ONLY + SIDE_EFFECTS

    def __init__(self, repo: InMemoryPlacementRepo, notifier, clock=lambda: datetime.now(timezone.utc)):
        self.repo = repo
        self.notifier = notifier
        self.clock = clock

    def functions(self) -> dict:
        return {name: getattr(self, name) for name in self.TOOL_NAMES}

    def call(self, name: str, args: dict) -> dict:
        return dispatch(self.functions(), name, args)

    # ================================================================== SAMPLE 1 (given): read-only

    def _evaluate(self, s: Student, drive_id: int) -> list[dict]:
        # The business rule lives in data (placement.eligibility_rule), not in the prompt or an if.
        failed = []
        for rule in self.repo.rules_for_drive(drive_id):
            actual = getattr(s, rule.field)
            if not _passes(rule, actual):
                failed.append({"rule_id": rule.id, "rule": str(rule), "actual": actual})
        return failed

    def check_eligibility(self, student_id: str, drive_id: int) -> dict:
        """Decide whether ONE student may apply to ONE drive, using the drive's eligibility rules.

        Use before apply_to_drive, or when the user asks "can I apply", "am I eligible for
        <company>", or "why can't I apply". Do NOT use to find drives; use list_open_drives.
        Read-only: changes nothing.

        Args:
            student_id: Roll number, e.g. "22CS045".
            drive_id: Integer id returned by list_open_drives. Never a company name.

        Returns:
            {"student_id", "drive_id", "eligible", "failed_rules": [{"rule_id", "rule", "actual"}]}.
            Explain every failed rule to the user; do not invent rules that are not listed.
        """
        # Failures are returned, never raised: a stable code plus a hint telling the model what to do next.
        s = self.repo.get_student(student_id)
        if s is None:
            return _unknown_student(student_id)
        if self.repo.get_drive(drive_id) is None:
            return _unknown_drive(drive_id)
        # Every failed rule, not just the first: the model explains the verdict, it never decides it.
        failed = self._evaluate(s, drive_id)
        return {"student_id": s.roll_no, "drive_id": drive_id,
                "eligible": not failed, "failed_rules": failed}

    # ================================================================== SAMPLE 2 (given): side effect

    def apply_to_drive(self, student_id: str, drive_id: int) -> dict:
        """Submit a placement application for ONE student to ONE drive.

        Side effect: creates an application record the placement cell will act on. Call it only
        when the user clearly asks to apply or register ("apply me", "sign me up"), never to
        check or explore. Eligibility is re-checked here, but call check_eligibility first so
        you can explain the result.

        Args:
            student_id: Roll number, e.g. "22CS045".
            drive_id: Integer id returned by list_open_drives.

        Returns:
            {"application_id", "student_id", "drive_id", "status": "applied",
             "available_slots": [{"slot_id", "starts_at"}]}. Offer the slots to the user;
            book one only when they choose.
        """
        # Side effects check in a fixed order and stop at the first failure.
        s = self.repo.get_student(student_id)
        if s is None:
            return _unknown_student(student_id)
        d = self.repo.get_drive(drive_id)
        if d is None:
            return _unknown_drive(drive_id)
        if d.status != "open" or d.deadline <= self.clock():
            return {"error": "drive_closed",
                    "hint": f"{d.company} is not accepting applications. Call list_open_drives for open ones."}
        # Re-check even though the description says "call check_eligibility first".
        # An instruction asks; code enforces. The model may have skipped it.
        failed = self._evaluate(s, drive_id)
        if failed:
            return {"error": "not_eligible", "failed_rules": failed,
                    "hint": "Explain the failed rules to the user. Do not retry."}
        try:
            application_id = self.repo.create_application(s.id, drive_id)
        except AlreadyApplied:
            return {"error": "already_applied",
                    "hint": "The student has already applied to this drive. Tell the user; do not retry."}
        # Return what the next step needs: the model will want to offer interview slots.
        slots = self.repo.free_slots(drive_id)
        return {"application_id": application_id, "student_id": s.roll_no, "drive_id": drive_id,
                "status": "applied",
                "available_slots": [{"slot_id": sl.id, "starts_at": sl.starts_at.isoformat()} for sl in slots]}

    # ================================================================== YOUR TOOLS

    def get_student(self, student_id: str) -> dict:
        """Look up ONE student by roll number.

        Read-only. Use when the user asks "who is 22CS045", "show my details", or "what is my CGPA".
        Do NOT use it for eligibility or applications; use check_eligibility or apply_to_drive instead.

        Args:
            student_id: Roll number, e.g. "22CS045".

        Returns:
            {"student_id", "name", "branch", "cgpa", "backlogs", "grad_year"}.
            student_id in the result is the roll number. Error: {"error": "unknown_student", "hint": "..."}
            if not found.
        """
        s = self.repo.get_student(student_id)
        if s is None:
            return _unknown_student(student_id)
        return {"student_id": s.roll_no, "name": s.name, "branch": s.branch,
                "cgpa": s.cgpa, "backlogs": s.backlogs, "grad_year": s.grad_year}

    def list_open_drives(self, branch: str | None = None, grad_year: int | None = None) -> dict:
        """List all placement drives that are currently open.

        Read-only. Use when the user asks "what companies are hiring", "show me open drives",
        or "which drives can CSE 2026 students apply to". Do NOT use for eligibility of one student;
        use check_eligibility instead. The repository already returns only open drives, sorted by
        soonest deadline first.

        Args:
            branch: Optional branch filter, e.g. "CSE". If given, exclude drives whose branch rule
                    rejects that branch.
            grad_year: Optional graduation year filter, e.g. 2026. If given, exclude drives whose
                       grad_year rule rejects that year.

        Returns:
            {"drives": [{"drive_id", "company", "role", "ctc_lpa", "deadline"}]}.
            Deadline formatted as YYYY-MM-DD. Only drives with status "open" and deadline after now,
            sorted by soonest deadline first.
        """
        drives = []
        for d in self.repo.list_open_drives(self.clock()):
            rules = self.repo.rules_for_drive(d.id)
            skip = False
            for r in rules:
                if r.field == "branch" and branch is not None and not _passes(r, branch):
                    skip = True
                if r.field == "grad_year" and grad_year is not None and not _passes(r, grad_year):
                    skip = True
            if not skip:
                drives.append({"drive_id": d.id, "company": d.company, "role": d.role,
                               "ctc_lpa": d.ctc_lpa, "deadline": d.deadline.date().isoformat()})
        return {"drives": drives}

    def book_interview_slot(self, student_id: str, slot_id: int) -> dict:
        """Book ONE interview slot for ONE student.

        Side effect: reserves a slot so the company can interview the student. Call only when the user
        clearly chooses a slot, such as "book me at 10 AM". Do NOT use this to list slots; use
        apply_to_drive first to get valid slots.

        Args:
            student_id: Roll number, e.g. "22CS045".
            slot_id: Integer id of the slot offered by apply_to_drive.

        Returns:
            {"slot_id", "drive_id", "starts_at", "status": "booked"}.
            starts_at is ISO-8601. Errors:
            - unknown_student
            - unknown_slot
            - no_application (student never applied to that drive)
            - slot_taken (include remaining available_slots)
        """
        s = self.repo.get_student(student_id)
        if s is None:
            return _unknown_student(student_id)
        sl = self.repo.get_slot(slot_id)
        if sl is None:
            return {"error": "unknown_slot",
                    "hint": f"No slot with id {slot_id}. Use apply_to_drive to get valid slots."}
        if not self.repo.has_application(s.id, sl.drive_id):
            return {"error": "no_application",
                    "hint": "The student has not applied to this drive. Apply first."}
        if not self.repo.claim_slot(slot_id, s.id):
            slots = self.repo.free_slots(sl.drive_id)
            return {"error": "slot_taken",
                    "hint": "That slot is already booked. Offer remaining slots.",
                    "available_slots": [{"slot_id": x.id, "starts_at": x.starts_at.isoformat()} for x in slots]}
        return {"slot_id": sl.id, "drive_id": sl.drive_id,
                "starts_at": sl.starts_at.isoformat(), "status": "booked"}

    def notify_student(self, student_id: str, message: str) -> dict:
        """Queue a notification for ONE student.

        Side effect: sends a reminder, alert, or status update to the student's outbound communication
        channel. Use when the user asks to notify a student, such as "tell 22CS045 the interview is on
        Thursday" or "send a reminder about next week's drive". Do NOT use this to read or change
        academic or application data; use get_student or apply_to_drive instead.

        Args:
            student_id: Roll number, e.g. "22CS045".
            message: Exact text to send. It must be non-empty and no more than 160 characters.

        Returns:
            {"notification_id", "status": "queued"}. If the student is unknown, return the standard
            unknown_student error; if the message is empty or too long, return invalid_message.
        """
        s = self.repo.get_student(student_id)
        if s is None:
            return _unknown_student(student_id)
        if message is None or len(message.strip()) == 0 or len(message) > 160:
            return {"error": "invalid_message",
                    "hint": "Message must be non-empty and at most 160 characters long."}
        notification_id = self.notifier.send(student_id, message)
        return {"notification_id": notification_id, "status": "queued"}

    def list_my_applications(self, student_id: str) -> dict:
        """List ALL applications for ONE student, oldest first.

        Read-only. Use when the user asks "where have I applied", "what are my applications",
        or "when is my interview". Do NOT use this for choosing a drive or checking eligibility;
        use list_open_drives and check_eligibility for that. The repository already keeps the list,
        and the data includes both an application date and the booked interview time when one exists.

        Args:
            student_id: Roll number, e.g. "22CS045".

        Returns:
            {"applications": [{"application_id", "drive_id", "company", "role", "status",
            "applied_on", "interview_at"}]}. Applications are ordered oldest first, and interview_at
            is the booked slot time or None if no interview has been booked yet.
            Error: {"error": "unknown_student", "hint": "..."} if not found.
        """
        s = self.repo.get_student(student_id)
        if s is None:
            return _unknown_student(student_id)
        apps = []
        for app in self.repo.list_applications(s.id):
            apps.append({
                "application_id": app["application_id"],
                "drive_id": app["drive_id"],
                "company": app["company"],
                "role": app["role"],
                "status": app["status"],
                "applied_on": app["created_at"],
                "interview_at": app["interview_at"],
            })
        return {"applications": apps}
