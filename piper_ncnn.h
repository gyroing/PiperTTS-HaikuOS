#ifndef PIPER_NCNN_H_
#define PIPER_NCNN_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error Codes
#define PIPER_NCNN_OK 0
#define PIPER_NCNN_DONE 1
#define PIPER_NCNN_ERR_GENERIC -1
#define PIPER_NCNN_ERR_INVALID_INPUT -2
#define PIPER_NCNN_ERR_MODEL_LOAD_FAILED -3
#define PIPER_NCNN_ERR_SYNTHESIS_FAILED -4

/** @brief Opaque struct representing the synthesizer instance. */
typedef struct piper_ncnn_synthesizer piper_ncnn_synthesizer;

/** @brief Audio sample format. */
typedef int16_t piper_ncnn_audio_sample;

/**
 * @brief Options for synthesis.
 */
typedef struct piper_ncnn_synthesize_options {
  float length_scale;        /**< How fast the text is spoken (>1 faster, <1 slower). */
  float noise_scale;         /**< Controls how much noise is added during synthesis. */
  float noise_w_scale;       /**< Controls how much phonemes vary in length. */
} piper_ncnn_synthesize_options;

/**
 * @brief Struct to hold a chunk of audio data.
 */
typedef struct {
    const piper_ncnn_audio_sample *samples;
    size_t num_samples;
} piper_ncnn_audio_chunk;


/**
 * @brief Reads the default synthesis options directly from the loaded synthesizer instance.
 * @param synth The synthesizer instance.
 * @return An options struct with the default values for that specific model.
 */
piper_ncnn_synthesize_options piper_ncnn_get_default_options(piper_ncnn_synthesizer *synth);

/**
 * @brief Creates a synthesizer instance, loading its model and config.
 * @param model_dir Path to the models directory.
 * @param language Language code to find the model files.
 * @param speaker_id The ID of the speaker.
 * @return A pointer to the synthesizer instance, or NULL on error.
 */
piper_ncnn_synthesizer *piper_ncnn_create(const char *model_dir, const char* language, int speaker_id);

/**
 * @brief Frees all resources associated with a synthesizer instance.
 * @param synth The synthesizer instance to free.
 */
void piper_ncnn_free(piper_ncnn_synthesizer *synth);

/**
 * @brief Converts an error code into a human-readable string.
 * @param error_code The error code returned by a function.
 * @return A constant string describing the error.
 */
const char* piper_ncnn_strerror(int error_code);

/**
 * @brief Starts the synthesis process for a line of phonemes and generates the full audio into an internal buffer.
 * @param synth The synthesizer instance.
 * @param phonemes A string of phonemes to synthesize.
 * @param options Synthesis options. If NULL, the model's defaults will be used.
 * @return PIPER_NCNN_OK on success.
 */
int piper_ncnn_synthesize_start(piper_ncnn_synthesizer *synth, const char *phonemes, const piper_ncnn_synthesize_options *options);

/**
 * @brief Reads the next chunk of synthesized audio from the internal buffer.
 * @param synth The synthesizer instance.
 * @param chunk A pointer to a chunk struct to be filled.
 * @return PIPER_NCNN_OK if a valid chunk was read, PIPER_NCNN_DONE if all audio has been read.
 */
int piper_ncnn_synthesize_next(piper_ncnn_synthesizer *synth, piper_ncnn_audio_chunk *chunk);

/**
 * @brief (For compatibility/CLI tools) Synthesizes a line and writes the audio directly to stdout.
 * @param synth The synthesizer instance.
 * @param phonemes A string of phonemes to synthesize.
 * @param options Synthesis options. If NULL, the model's defaults will be used.
 * @return PIPER_NCNN_OK on success.
 */
int piper_ncnn_synthesize_line(piper_ncnn_synthesizer *synth, const char *phonemes, const piper_ncnn_synthesize_options *options);


#ifdef __cplusplus
}
#endif

#endif // PIPER_NCNN_H_

