<div align="center">

# ACTUS

**Autonomous Crew Tracking and Utility System**  
An on-board assistant for astronauts running scientific experiments in space.

![status](https://img.shields.io/badge/status-in%20development-lightgrey?style=flat-square)
![offline](https://img.shields.io/badge/operation-fully%20offline-blue?style=flat-square)
![purpose](https://img.shields.io/badge/purpose-BAS%20%2F%20lunar%20missions-black?style=flat-square)

</div>

<br>

<div align="center">
<i>Ground control is far away. Actus watches the experiment so no one has to.</i>
</div>

<br>

---

## What it is

Actus is a human activity recognition system built for astronauts conducting experiments on space stations and future lunar missions. When ground support is minutes or hours away due to communication delay, Actus stays with the astronaut, step by step, in real time.

It watches a scientific experiment unfold through fixed cameras already installed on the payload, and it understands what step is happening, what step comes next, and whether anything has gone out of order.

## What it does

**Follows the sequence.** Actus tracks the astronaut's movements and the experiment's progress against the protocol it was trained on, step by step, without needing a live link to Earth.

**Tells you what's next.** At the start of the experiment and after every completed step, Actus prompts the astronaut with the next action to take.

**Catches mistakes as they happen.** If a step is skipped or performed out of order, Actus raises a voice alert immediately, so the astronaut's hands stay free and their eyes stay on the work.

**Keeps the record.** Every step is logged with a timestamp and an outcome, written into a lightweight text file that documents exactly what happened and when.

**Sends and saves the footage.** The experiment is streamed live to a designated address on the ground and stored locally at the same time, so nothing is lost if the connection drops.

**Shows it all in one place.** A graphical interface brings the live feed, the current step, and the alert history together, so anyone monitoring the experiment can see its state at a glance.

## Why it matters

Bandwidth in space is scarce and communication delay is unavoidable. Raw video cannot be streamed to Earth for every experiment, and mission control cannot watch every second of every protocol. Actus moves that oversight on board, running entirely at the edge, so an experiment can be executed correctly even when no one on Earth is watching it happen.

<br>

<div align="center">

—

*Built for missions where the nearest expert is hundreds of thousands of kilometers away.*

</div>
