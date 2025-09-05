# PiperTTS for Haiku Based on NCNN Models
Natural human voice  offline tts (piper tts)
Usage: pipertts -m <model_name> [OPTIONS]

A command-line Text-to-Speech (TTS) program for the Haiku OS using Piper NCNN models.
The program automatically detects the model configuration from the specified directory.
Input text is read with the following priority: -t > -f > stdin.

## Mandatory Argument:
  -m, --model <model_name>  The name of the model directory to use.

## Options:
  -p, --path <path>         Path to the base directory containing model folders.
                            (Default: /boot/home/config/non-packaged/data/piperncnn/models/)

  -s, --speaker_id <id>     The integer ID of the speaker/voice to use.
                            (Default: 0).

  -l, --lang <lang_code>    Override the espeak voice used for phoneme generation
                            (e.g., "fr-fr"). If not provided, the "espeak_voice" value
                            from the config file is used.

  -t, --text "..."          The text string to synthesize. Highest priority input.

  -f, --file <path>         Read input text from a UTF-8 file.

  -o, --output <path>       Path to save the output as a WAV file. If omitted, the
                            synthesized audio will be played directly.

  -h, --help                Show this help message and exit.

--------------------------------------------------------------------------------

## Download Models from:
** https://huggingface.co/gyroing/PiperTTS-NCNN-Models **

