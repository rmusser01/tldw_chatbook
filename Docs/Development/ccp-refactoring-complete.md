# CCP Screen Refactoring - Complete Documentation

## Overview

This document records the current Personas destination architecture and its retained CCP support code.

## Architecture Overview

### Component Hierarchy

```
PersonasScreen (CCP destination)
├── Handlers (Business Logic)
│   ├── CCPCharacterHandler
│   ├── CCPPersonaHandler
│   └── destination-specific controllers and workers
└── Shared CCP support code
    ├── ccp_messages.py (active character message definitions)
    ├── CCPMessageManager (compatibility/test-only; not constructed)
    └── validators, loading indicators, and decorators
```

## Core Components

### 1. PersonasScreen (`personas_screen.py`)

The main Personas destination owns character and persona browsing, editing, previews, and the launch/handoff path into main chat.

**Key Responsibilities:**
- Constructs only `CCPCharacterHandler` and `CCPPersonaHandler`
- Routes destination-native character and persona events
- Coordinates its conversation and preview controllers for the selected persona or character
- Hands launch/navigation requests to the main application

### 2. Handler and shared support modules

The live screen handler set is `CCPCharacterHandler` and `CCPPersonaHandler`. `ccp_messages.py` remains active through `CharacterMessage.Loaded`; validators, loading indicators, and decorators remain shared helpers. `CCPMessageManager` is retained compatibility/test-only support, not a constructed screen handler. The retired conversation and dictionary handlers had no production construction path and are not part of the current architecture.

## Message System

### Message Flow Architecture

```
User Action → Widget → Message → Screen → Handler → Worker → Database
                ↑                     ↓
                └── UI Update ← Message ← call_from_thread
```

`PersonasScreen` routes destination-native character and persona messages, including `CharacterMessage.Loaded`. Shared CCP message definitions may remain for compatibility, but they do not create a live conversation, prompt, or dictionary handler path.
