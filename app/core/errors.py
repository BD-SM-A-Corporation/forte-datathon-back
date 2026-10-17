class AnalysisError(Exception):
    """Safe, fixed messages that may be returned to API clients."""


class VideoError(AnalysisError):
    pass


class ConfigurationError(AnalysisError):
    pass


class InferenceError(AnalysisError):
    pass


class BudgetExceeded(AnalysisError):
    pass
