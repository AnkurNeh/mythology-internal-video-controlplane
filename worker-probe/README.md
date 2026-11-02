# SceneCraft Worker Probe

This probe contains no permanent secrets.

It performs a short diagnostic:
1. exchanges the one-time SceneCraft launch binding for a short-lived worker session;
2. authenticates back to SceneCraft;
3. verifies read access to the durable queue;
4. exits without claiming customer work, loading a model, generating video, or charging a customer wallet.

The Hugging Face job receives only non-secret launch bindings as environment variables.
