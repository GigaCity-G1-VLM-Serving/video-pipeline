"""Stand-ins for C (serving) and D (scoring), used only for A/B testing.

These are not C's or D's implementations. Results from the mock server are marked
``is_synthetic``; results through the vLLM shim come from an A/B test server, not
from C's serving measurements.
"""
