"""Compatibility shim: the provider tests live in test_scheduler_providers.py."""

from tests.test_scheduler_providers import SchedulerProviderTests as SchedulerProviderTests

SchedulerProviderTests.__test__ = False
