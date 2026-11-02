"""Telegram channel package.

The package marker stays inert: importing it must not pull in the runtime or a
platform SDK, because discovery imports ``manifest`` for every channel.
"""
