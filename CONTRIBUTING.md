# Contributing

## Scope

This repository is primarily a robotics portfolio project, but contributions are still easier to review when they follow a small amount of structure.

## Good Contribution Areas

- bug fixes in the main runtime
- calibration or setup improvements
- test cleanup for stale scripts and missing imports
- documentation corrections
- dashboard usability improvements

## Before Opening a Change

- explain what hardware setup you tested on
- mention whether the change affects `main.py`, a tuning tool, or firmware
- call out any assumptions about camera format, model files, or serial device paths

## Safety Expectations

- treat motor changes as safety-sensitive
- describe bench-test steps for anything that changes motion behavior
- prefer low-speed defaults in examples

## Documentation Expectations

- keep docs aligned with the code that actually exists
- mark experimental or model-dependent features clearly
- avoid claiming performance numbers that are not measured in this repo

## Suggested Pull Request Checklist

- code builds or at least imports cleanly in the changed area
- documentation was updated if behavior changed
- hardware assumptions are stated explicitly
- stale or legacy behavior was called out rather than hidden
