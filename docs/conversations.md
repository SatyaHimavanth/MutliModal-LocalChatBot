# Conversation timelines

- A conversation has an ordered message stream. Editing a saved user message sends a new turn in a child timeline.
- The child inherits the parent timeline strictly before the edited user message. The edited turn and the assistant response that followed it are omitted from the child; the new edited user message and response begin its own message stream.
- Parent and sibling timelines remain unchanged. The sidebar lists the root conversation once; an in-chat previous/next control switches between its saved timelines.
- The edit starts with the collection selection saved for the original user turn. Users can adjust collections before sending; the edited turn persists the selection actually used.
- User-message attachments and extracted text are encrypted with the message and restored on its timeline. Editing a turn carries its attachments into the new branch.
- Timeline ancestry is represented by parent conversation ID and forked message ID. Reads reconstruct the visible transcript recursively, with a 32-level ancestry cap.
- Conversation actions include pin, rename, archive/restore, and delete. Deleting a conversation removes all of its timelines. Token-aware history pruning, regeneration, and LangGraph checkpointing are not implemented. Rust reconstructs the ancestor message list directly from SQLite.
- Deleting a user turn removes that turn and all later turns from the selected timeline. Removing an assistant response removes it and all later turns, leaving its user prompt available for regeneration. These actions update the selected timeline in place; edited sibling timelines remain unchanged. Hidden message rows are retained so other existing timeline branches can still resolve shared ancestry.
- Sending saves the user turn and a pending assistant response before retrieval or generation starts. Returning to that conversation during generation restores the thinking state and current in-memory stream; its root chat shows an activity spinner in the sidebar. Stop cancels the model stream and saves partial output plus “User interrupted generation.” A pending turn left by app exit is changed to an interrupted state at next startup.
