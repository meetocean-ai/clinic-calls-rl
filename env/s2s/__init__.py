"""Speech-to-speech policy adapters (roadmap P2-06), code only until a GPU run (P2-11): each takes the env's caller
audio, drives one served S2S model, routes its tool calls through the env's tools and answers with the model's own
audio. Every adapter runs offline against a stub server (`env/s2s/test_adapters.py`); the transports that talk to the
real servers are bound and verified when the models are first served.

    omni.py          Qwen3-Omni behind vLLM-Omni: OpenAI-compatible chat with audio input, function calling, speech out
    voicechat.py     NemotronLabs VoiceChat: full-duplex stream with a separate <TOOLCALL> channel and on-hold lines
    personaplex.py   PersonaPlex / Moshi: full-duplex, no tool calling of its own → delegation to a text LLM
    transport.py     the event protocol the duplex adapters speak, and a stub-friendly WebSocket binding
"""
