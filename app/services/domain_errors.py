class DomainError(RuntimeError):
    """Validated business rejection; preserve through the SQL unit of work."""
    domain_error = True


class FeedbackConflict(DomainError):
    pass


class FeedbackValidation(DomainError):
    pass


class FeedbackUnavailable(DomainError):
    pass
