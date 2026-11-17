# UI Reference Notes

The user supplied two screenshots of the llama.cpp web UI as visual references. They show a restrained, light chat layout with a left conversation sidebar, a centered composer, a model selector inside the composer, and compact per-message performance and branch controls. These screenshots are references for interaction placement and information density, not a request to clone llama.cpp branding or model management.

## Confirmed adaptations

- Put a **collections selector** in the composer where the reference UI places its model selector. It supports all, none, or a chosen subset of the employee's collections.
- When the default-off `mcp-demo` build feature is enabled, put the MCP connector selector beside the attachment button. The selector lists connection availability and lets users opt connectors into the current conversation.
- Keep connector administration on a separate page; when the build feature is disabled, remove its navigation, composer control, page, and dialog from the UI.
- New chats select all collections by default.
- If a user changes collections during a chat, warn that future turns will retrieve from a different context. Apply the change only to future turns.
- Editing a previous user input creates a new timeline. Show the branch position as `<x/y>` near/beneath the edited user input. Preserve the old timeline.
- If the user changes collections while editing an earlier input, the new timeline uses the newly selected collection set; otherwise it inherits the set from the fork point.
- Show per-turn input/output token totals, elapsed time, and input/output throughput. Keep these details close to the associated user/assistant turn and make their meaning clear (prompt processing vs. generated output). While pending, show the thinking or current tool step without an ETA.
- Make retrieved sources/citations and tool progress inspectable without crowding the default chat view. Show a prominent current tool step and a collapsible activity log; collapse it when the final answer begins streaming. Persist the safe tool log with its assistant message. Keep raw tool arguments and response bodies hidden, showing only names, status, timing, and safe summaries.

## Reference image handling

The screenshots were provided in conversation and are not currently stored as files in the repository. These written notes preserve the decisions that future UI work needs. If the image files are later added, link them here without copying large files into source control unnecessarily.
