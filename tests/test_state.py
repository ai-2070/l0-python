"""Tests for l0.state module."""

import time

import pytest

from l0.state import append_token, create_state, mark_completed, update_checkpoint
from l0.types import State


class TestCreateState:
    def test_creates_fresh_state(self):
        state = create_state()
        assert state.content == ""
        assert state.token_count == 0
        assert state.completed is False


class TestUpdateCheckpoint:
    def test_saves_checkpoint(self):
        state = State(content="Hello world")
        update_checkpoint(state)
        assert state.checkpoint == "Hello world"


class TestAppendToken:
    def test_appends_token(self):
        state = State()
        append_token(state, "Hello")
        assert state.content == "Hello"
        assert state.token_count == 1

    def test_multiple_tokens(self):
        state = State()
        append_token(state, "Hello")
        append_token(state, " ")
        append_token(state, "World")
        assert state.content == "Hello World"
        assert state.token_count == 3

    def test_sets_first_token_at(self):
        state = State()
        append_token(state, "a")
        assert state.first_token_at is not None

    def test_updates_last_token_at(self):
        state = State()
        append_token(state, "a")
        first_time = state.last_token_at
        assert first_time is not None

        time.sleep(0.01)
        append_token(state, "b")

        assert state.last_token_at is not None
        assert state.last_token_at > first_time


class TestMarkCompleted:
    def test_marks_completed(self):
        state = State()
        mark_completed(state)
        assert state.completed is True

    def test_calculates_duration(self):
        state = State()
        state.first_token_at = time.time() - 1.0
        state.last_token_at = time.time()

        mark_completed(state)

        assert state.duration is not None
        assert state.duration >= 0.9
