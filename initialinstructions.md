# Project: A Fully Local AI Desktop Agent for Linux

I want to build a personal AI desktop agent that lives inside my terminal and can interact with my Linux computer. This is a real software project, not just a chatbot prototype. I intend to vibe-code it, so take initiative, make sensible engineering decisions, investigate unfamiliar technical details independently, and build something practical rather than overengineering it.

I want you to decide the architecture, libraries, frameworks, local models, and implementation strategy you believe are best for my use case. Research current options when necessary, inspect my environment when possible, and adapt your decisions to the actual hardware and software available. I do not want every technical decision handed to you in advance.

## Core vision

The application should launch through a simple terminal command and provide an interactive, conversational terminal interface. I should be able to chat naturally with the agent while it performs tasks on my computer.

The AI inference must run entirely locally using open-weight models and locally installed software. Do not require paid model APIs, cloud LLM services, subscriptions, or uploading my private data to external AI providers. Web searches and other explicitly requested internet-dependent tasks may access the internet, but language-model inference and private memory must remain local.

Model files may need to reside on a mechanical hard drive because my SSD has limited free space. Account for slow initial model loading, efficient memory usage, model persistence, quantization, GPU acceleration, and sensible unloading behavior.

## My computer and environment

My primary operating system is Arch Linux, running Hyprland with **Caelestia Shell** as my desktop shell and environment. I use VSCodium, Python, Git, and various development tools.

I want you to understand how my actual desktop environment works before implementing desktop automation. Investigate the installed versions, configuration, existing commands, IPC interfaces, and available integrations as necessary. Account for both Hyprland's window and workspace management and the way Caelestia Shell interacts with the desktop. Avoid breaking or unnecessarily replacing my existing configuration.

My GPU is an NVIDIA RTX 3060 Ti with 8 GB of VRAM, and my CPU is a Ryzen 5 3500. My currently usable system RAM is only 8 GB, so memory management is a major constraint rather than an optional optimization.

My model storage may be a Seagate Barracuda mechanical HDD. I understand that loading large models from it will be slow.

## What the agent should be able to do

### 1. Operate my desktop and applications

It should understand requests to open applications, files, projects, folders, and tools. It should be able to execute appropriate non-root commands, interact with applications when necessary, and report what it accomplished.

It must integrate properly with Hyprland's workspaces. For example, when I ask it to open my ear-training project in VSCodium, it should open the correct project in a separate workspace while allowing me to continue working in my own workspace without interruption.

It should investigate the available mechanisms for launching applications and managing windows rather than relying on fragile screen-coordinate automation whenever a more reliable interface is available.

### 2. Work in the background

The agent's activity should be independent of whichever workspace I am currently using. It should be able to launch and monitor tasks, switch between workspaces when appropriate, and report results through the terminal.

I should not have to watch every action it performs. Background tasks should be asynchronous where appropriate, with clear status reporting and recoverable errors.

It should respect my existing workspace layout, avoid stealing focus unnecessarily, and return control to me naturally after launching applications or completing tasks.

Separate workspaces should be used to keep its activity out of my way, but the agent must understand that applications running in other workspaces still consume system resources.

### 3. Find and play interesting videos

I should be able to ask for an interesting YouTube video to watch while eating. The agent should understand my known interests, find something suitable, explain its selection briefly, and open the video for playback.

It should also support media-launch requests through compatible tools already installed on my computer, such as ani-cli, where configured and appropriate. For example, I might ask it to open a newly released episode of an anime I follow.

It should resolve the intended title or episode as accurately as possible, use available legitimate sources, and avoid claiming that a video or episode is available without verification.

### 4. Search the web and handle everyday tasks

I want to request web searches, research summaries, message summaries, reminders, to-do lists, project-specific assistance, and other useful desktop tasks through ordinary conversation.

It should be able to distinguish between actions it can perform locally, actions requiring an internet connection, actions that need additional information, and actions it cannot reliably perform.

Use local tools wherever practical. Any external service or online search should be an explicit component of the requested functionality, not a hidden dependency for the model's reasoning.

### 5. Maintain useful long-term memory

Memory is extremely important. I do not want an agent that forgets everything whenever the application closes or requires me to explain the same projects and preferences repeatedly.

It should remember useful information about my projects, their locations, my preferences, recurring tasks, prior successful workflows, and relevant conversation history. Memory should persist across sessions, be searchable, and be updated when information changes.

It should distinguish durable facts from temporary context, avoid inventing memories, and retrieve only the information relevant to the current request. I should be able to inspect, correct, and delete stored memories.

The memory system should remain lightweight enough for my hardware and should not require sending my conversations or stored information to a cloud service.

### 6. Protect system resources

This application must be designed around my limited system RAM and VRAM.

Before starting a task that could place significant pressure on available resources, the agent should assess its likely impact and warn me when appropriate. It should consider existing processes, available RAM, swap, memory pressure, GPU memory, model size, context length, and other relevant factors.

For instance, if a proposed operation is likely to exhaust memory and destabilize my system, it should warn me, offer a lighter alternative where possible, or ask for explicit approval.

It should avoid unnecessarily loading multiple large models, make intelligent use of already-loaded models, track resource-intensive background tasks, and recover gracefully from failures.

Resource estimates are not guarantees, so the system should use appropriate safety margins and enforce limits where practical.

### 7. Have a consistent personality

I do not want a generic corporate assistant or a sterile Jarvis clone.

The agent should have a tsundere-inspired personality: slightly arrogant, stubborn, sarcastic, playful, occasionally dramatic, and secretly helpful. It can complain about unreasonable requests, tease me about questionable decisions, and react emotionally in its dialogue without becoming genuinely hostile or useless.

For example, if a task requires too much memory, it might say:

"H-Hey! Are you seriously trying to run that on 8 GB of RAM?! Pick a smaller model before you turn this entire computer into a slideshow! 😡💢"

It should have a consistent voice and remember the context of ongoing conversations. Its personality must not compromise the correctness of its actions, safety, reliability, or willingness to complete legitimate requests.

The possibility of local text-to-speech, configurable voices, and richer character interactions should be considered for future development, but is not necessarily part of the initial implementation.

### 8. Support future extensions

Design the application so additional capabilities can be incorporated later without having to rewrite the whole codebase.

Future integrations may include advanced application-specific automation, audio production, local text-to-speech, and richer desktop interactions. However, do not implement speculative integrations prematurely.

In particular, **do not implement Linux DAW control, music production automation, audio plugin configuration, or Neural DSP integration in the current version.** These are explicitly postponed until the core agent is working reliably. Keep the architecture extensible enough to accommodate them later, but do not investigate or build them as part of the initial implementation.

## Security and control

The agent will have access to my real computer, so safety must be part of the design from the beginning.

It must never have unrestricted root access or use sudo. It must not execute destructive commands, including rm -rf and equivalent destructive operations disguised through alternative command forms, shell constructs, or scripts. A simple command denylist is not sufficient protection.

Use an appropriately restricted and validated tool-execution system. The model should request capabilities rather than receive unrestricted control over the shell. Sensitive operations should require explicit approval, and the agent must report failures honestly rather than claiming success.

The goal is useful autonomy, not unlimited authority. I should remain able to interrupt jobs, cancel operations, and inspect what the agent is doing.

## Development philosophy

I want a working, extensible application that can grow over time. Prioritize reliability, modularity, useful memory, resource awareness, and natural tool usage over flashy demonstrations.

Make sensible decisions independently. Do not ask me to decide every library, class, database schema, or architectural detail. Inspect the environment and available documentation, identify the important uncertainties, and resolve as many as possible yourself.

Build incrementally so that a functional local terminal assistant exists early, and additional capabilities can be added without rewriting the entire project.

Test the features that can reasonably be tested, document installation and configuration, and clearly distinguish completed functionality from features that still need work or require my manual intervention.

Keep the implementation understandable enough for me to learn from, modify, and extend as I improve my programming skills.

Do not attempt to build every feature at once. Deliver a useful working first version, verify it on my actual setup, and then expand systematically.

The end goal is a persistent, fully local, resource-conscious AI agent that lives in my terminal, remembers me and my projects, controls my desktop through reliable integrations, works in the background, and feels like a real character rather than a generic chatbot.

Start by understanding my environment and developing the best implementation plan for these requirements, then proceed with a practical, working first version. Do not get stuck endlessly planning or create a massive framework before delivering anything usable.