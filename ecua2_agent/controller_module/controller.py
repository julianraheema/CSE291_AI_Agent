#!/usr/bin/env python3

import time
import shlex
import logging
from typing import List, Tuple, Dict, Optional
import argparse

import pyautogui

# Safety: moving mouse to top-left corner aborts automation
pyautogui.FAILSAFE = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class ControllerError(Exception):
    """Custom exception for controller errors."""
    pass


class Controller:
    def __init__(self, default_wait: float = 0.2):
        self.default_wait = default_wait
    

    def execute_action(self, comd: str) -> str:
        comd = comd.strip()
        if not comd or comd.startswith("#"):
            return "Not Valid"

        action, args = self._parse_line(comd)
        action = action.upper()

        logger.info("Executing: %s %s", action, " ".join(args))

        if action == "MOVE_TO":
            self._move_to(args)
        elif action == "CLICK":
            self._click(args)
        elif action == "MOUSE_DOWN":
            self._mouse_down(args)
        elif action == "MOUSE_UP":
            self._mouse_up(args)
        elif action == "RIGHT_CLICK":
            self._right_click(args)
        elif action == "DOUBLE_CLICK":
            self._double_click(args)
        elif action == "DRAG_TO":
            self._drag_to(args)
        elif action == "SCROLL":
            self._scroll(args)
        elif action == "TYPING":
            self._typing(args)
        elif action == "PRESS":
            self._press(args)
        elif action == "KEY_DOWN":
            self._key_down(args)
        elif action == "KEY_UP":
            self._key_up(args)
        elif action == "HOTKEY":
            self._hotkey(args)
        elif action == "WAIT":
            self._wait(args)
        elif action == "FAIL":
            logger.warning("FAIL action received")
            return "FAIL"
        elif action == "DONE":
            logger.info("DONE action received")
            return "DONE"
        else:
            raise ControllerError(f"Unknown action: {action}")

        return "OK"

    # Execute list of actions for faster agent
    def execute_actions(self, lines: List[str]) -> str:
        for line in lines:
            status = self.execute_action(line)
            if status in ("FAIL", "DONE"):
                return status
        return "OK"


    @staticmethod
    def _parse_line(line: str) -> Tuple[str, List[str]]:

        # shlex for string with qoutes
        tokens = shlex.split(line)
        if not tokens:
            raise ControllerError("Empty command line")

        action = tokens[0]
        args = tokens[1:]
        return action, args


    @staticmethod
    def _parse_button(arg: Optional[str]) -> str:
        """
        Map button argument to pyautogui format.
        Defaults to 'left' if None.
        """
        if not arg:
            return "left"
        b = arg.lower()
        if b in ("left", "right", "middle"):
            return b
        raise ControllerError(f"Invalid mouse button: {arg}")

    @staticmethod
    def _parse_int_pair(args: List[str], offset: int = 0) -> Tuple[int, int]:
        """
        Parse two consecutive integers (e.g. x y) from args starting at offset.
        """
        try:
            x = int(args[offset])
            y = int(args[offset + 1])
            return x, y
        except (IndexError, ValueError):
            raise ControllerError(f"Expected two integers at positions {offset},{offset+1}, got: {args}")

    def _move_to(self, args: List[str]) -> None:
        if len(args) != 2:
            raise ControllerError("MOVE_TO requires exactly 2 arguments: x y")
        x, y = self._parse_int_pair(args)
        pyautogui.moveTo(x, y)

    def _click(self, args: List[str]) -> None:
        """
        CLICK [button] [x y] [num_clicks]

        Examples:
        CLICK
        CLICK right
        CLICK 100 200
        CLICK right 100 200 2
        """
        button = "left"
        x = y = None
        clicks = 1

        idx = 0
        # First arg could be button or x
        if idx < len(args):
            if args[idx].lower() in ("left", "right", "middle"):
                button = self._parse_button(args[idx])
                idx += 1

        # Next two could be x, y
        if idx + 1 < len(args):
            try:
                x = int(args[idx])
                y = int(args[idx + 1])
                idx += 2
            except ValueError:
                pass

        # Next could be num_clicks
        if idx < len(args):
            try:
                clicks = int(args[idx])
            except ValueError:
                raise ControllerError(f"Invalid num_clicks: {args[idx]}")

        if x is None or y is None:
            pyautogui.click(button=button, clicks=clicks)
        else:
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)

    def _mouse_down(self, args: List[str]) -> None:
        button = self._parse_button(args[0] if args else None)
        pyautogui.mouseDown(button=button)

    def _mouse_up(self, args: List[str]) -> None:
        button = self._parse_button(args[0] if args else None)
        pyautogui.mouseUp(button=button)

    def _right_click(self, args: List[str]) -> None:
        x = y = None
        if len(args) >= 2:
            x, y = self._parse_int_pair(args)
        if x is None or y is None:
            pyautogui.click(button="right")
        else:
            pyautogui.click(x=x, y=y, button="right")

    def _double_click(self, args: List[str]) -> None:
        x = y = None
        if len(args) >= 2:
            x, y = self._parse_int_pair(args)
        if x is None or y is None:
            pyautogui.click(clicks=2)
        else:
            pyautogui.click(x=x, y=y, clicks=2)

    def _drag_to(self, args: List[str]) -> None:
        if len(args) != 2:
            raise ControllerError("DRAG_TO requires exactly 2 arguments: x y")
        x, y = self._parse_int_pair(args)
        # Drag with left button pressed
        pyautogui.dragTo(x, y, button="left")

    def _scroll(self, args: List[str]) -> None:
        """
        SCROLL dx dy

        dy -> vertical scroll (positive up, negative down)
        dx -> horizontal scroll (if available)
        """
        if len(args) != 2:
            raise ControllerError("SCROLL requires exactly 2 arguments: dx dy")
        dx, dy = self._parse_int_pair(args)
        # Vertical scroll
        if dy != 0:
            pyautogui.scroll(dy)
        # Horizontal scroll (only in newer pyautogui)
        if dx != 0 and hasattr(pyautogui, "hscroll"):
            pyautogui.hscroll(dx)

    def _typing(self, args: List[str]) -> None:
        if not args:
            raise ControllerError("TYPING requires a text argument")
        # After shlex, quoted string is a single arg
        text = " ".join(args)
        pyautogui.typewrite(text)

    def _press(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ControllerError("PRESS requires exactly 1 key")
        key = args[0]
        pyautogui.press(key)

    def _key_down(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ControllerError("KEY_DOWN requires exactly 1 key")
        key = args[0]
        pyautogui.keyDown(key)

    def _key_up(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ControllerError("KEY_UP requires exactly 1 key")
        key = args[0]
        pyautogui.keyUp(key)

    def _hotkey(self, args: List[str]) -> None:
        """
        HOTKEY ctrl+alt+t
        or
        HOTKEY ctrl alt t
        """
        if not args:
            raise ControllerError("HOTKEY requires at least one key")

        if len(args) == 1:
            keys = args[0].split("+")
        else:
            keys = args

        keys = [k.strip() for k in keys if k.strip()]
        if not keys:
            raise ControllerError("HOTKEY parsed empty key list")

        pyautogui.hotkey(*keys)

    def _wait(self, args: List[str]) -> None:
        if args:
            try:
                seconds = float(args[0])
            except ValueError:
                raise ControllerError(f"Invalid WAIT time: {args[0]}")
        else:
            seconds = self.default_wait
        time.sleep(seconds)


# ----------- Example CLI usage ----------- #

def run_script_file(path: str) -> None:
    """
    Run a plain text script file where each line is an action.
    """
    ctrl = Controller()
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    status = ctrl.execute_actions(lines)
    print(f"Script finished with status: {status}")


if __name__ == "__main__":
    demo_actions = [
        "MOVE_TO 500 500",
        "WAIT 0.5",
        "CLICK",
        'TYPING "hello from controller"',
        "WAIT 0.5",
        "DONE",
    ]
    Controller().execute_actions(demo_actions)
