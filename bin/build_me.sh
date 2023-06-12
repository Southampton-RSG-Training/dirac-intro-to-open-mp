#!/usr/bin/env bash
set -e -x
python3 bin/get_schedules.py
python3 bin/get_setup.py
