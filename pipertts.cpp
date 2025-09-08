// ============================================================================
// pipertts.cpp
//
// A command-line Text-to-Speech (TTS) application for the Haiku operating system.
// It uses Piper NCNN models for speech synthesis and the native Haiku SoundPlayer
// API for audio playback. It relies on an external `espeak` executable for
// phoneme generation.
// ============================================================================

// C++ Standard Library Headers
#include <iostream>   // For console input/output (std::cout, std::cerr)
#include <fstream>    // For file stream operations (std::ifstream)
#include <sstream>    // For string stream operations (std::stringstream)
#include <string>     // For using the std::string class
#include <vector>     // For using the std::vector container
#include <queue>      // For std::queue, used for audio buffering
#include <mutex>      // For std::mutex, to ensure thread-safe access to the audio queue
#include <stdexcept>  // For standard exception types (std::runtime_error)
#include <cstdio>     // For popen, pclose
#include <algorithm>  // For std::algorithm functions
#include <cctype>     // For character handling functions
#include <dirent.h>   // For directory access (opendir, readdir)
#include <memory>     // For smart pointers (std::unique_ptr)

// Haiku OS specific headers
#include <SoundPlayer.h> // For the native Haiku audio playback API
#include <MediaDefs.h>   // For media-related definitions like audio formats
#include <unistd.h>      // For POSIX functions like usleep
#include <string.h>      // For memset

// Piper NCNN TTS engine header
#include "piper_ncnn.h"

// ============================================================================
// Helper Classes
// ============================================================================

/**
 * @struct WavWriter
 * @brief A helper class to create and write audio data to a WAV file.
 * It handles the complexities of the WAV file header.
 */
struct WavWriter {
    std::ofstream out;        // The output file stream.
    int samplerate, channels, bits; // Audio format properties.
    size_t data_bytes = 0;    // Tracks the total size of PCM data written.

    /**
     * @brief Constructs a WavWriter, opens the output file, and writes a placeholder header.
     * @param path The path to the output .wav file.
     * @param sr The sample rate of the audio (e.g., 22050).
     * @param ch The number of audio channels (e.g., 1 for mono).
     * @param b The number of bits per sample (e.g., 16).
     */
    WavWriter(const std::string &path, int sr, int ch, int b) : samplerate(sr), channels(ch), bits(b) {
        out.open(path, std::ios::binary);
        if (!out) throw std::runtime_error("Could not open output file for writing: " + path);
        // Write a 44-byte placeholder for the WAV header, which will be filled in later.
        for (int i = 0; i < 44; i++) out.put(0);
    }

    /**
     * @brief Writes a chunk of PCM audio data to the file.
     * @param pcm A vector of 16-bit audio samples.
     */
    void write(const std::vector<int16_t> &pcm) {
        out.write(reinterpret_cast<const char*>(pcm.data()), pcm.size() * sizeof(int16_t));
        data_bytes += pcm.size() * sizeof(int16_t);
    }

    /**
     * @brief Finalizes the WAV file by writing the correct header information and closing the file.
     * This must be called before the object is destroyed.
     */
    void close() {
        out.seekp(0); // Go back to the beginning of the file to write the final header.

        // Prepare WAV header fields
        int byte_rate = samplerate * channels * bits / 8;
        int block_align = channels * bits / 8;
        int subchunk2_size = data_bytes;
        int chunk_size = 36 + subchunk2_size;

        // Write the RIFF chunk descriptor
        out.write("RIFF", 4);
        out.write(reinterpret_cast<char*>(&chunk_size), 4);
        out.write("WAVE", 4);

        // Write the "fmt " sub-chunk
        out.write("fmt ", 4);
        int subchunk1_size = 16;
        out.write(reinterpret_cast<char*>(&subchunk1_size), 4);
        short audio_format = 1; // 1 for PCM
        out.write(reinterpret_cast<char*>(&audio_format), 2);
        short ch_short = channels;
        out.write(reinterpret_cast<char*>(&ch_short), 2);
        out.write(reinterpret_cast<char*>(&samplerate), 4);
        out.write(reinterpret_cast<char*>(&byte_rate), 4);
        short ba_short = block_align;
        out.write(reinterpret_cast<char*>(&ba_short), 2);
        short bps_short = bits;
        out.write(reinterpret_cast<char*>(&bps_short), 2);

        // Write the "data" sub-chunk
        out.write("data", 4);
        out.write(reinterpret_cast<char*>(&subchunk2_size), 4);

        out.close(); // Close the file stream.
    }
};


/**
 * @struct PiperEngine
 * @brief A C++ wrapper around the piper_ncnn C-style API for easier resource management (RAII).
 */
struct PiperEngine {
    piper_ncnn_synthesizer* synth = nullptr;

    // Destructor ensures that the synthesizer is freed automatically.
    ~PiperEngine() {
        if (synth) piper_ncnn_free(synth);
    }

    /**
     * @brief Initializes the synthesizer with a specified model.
     * @param model_dir The directory containing the NCNN model files.
     * @param lang The language code for the model (e.g., "fa").
     * @param voice_id The speaker ID to use from the model.
     */
    void init(const std::string& model_dir, const std::string& lang, int voice_id) {
        synth = piper_ncnn_create(model_dir.c_str(), lang.c_str(), voice_id);
        if (!synth) throw std::runtime_error("Failed to create Piper synthesizer.");
    }

    /**
     * @brief Synthesizes a sentence of phonemes into PCM audio data.
     * @param phoneme_sentence A string of IPA phonemes.
     * @return A vector of 16-bit PCM audio samples.
     */
    std::vector<int16_t> synthesize(const std::string& phoneme_sentence) {
        if (!synth || phoneme_sentence.empty()) return {};
        piper_ncnn_synthesize_options options = piper_ncnn_get_default_options(synth);
        if (piper_ncnn_synthesize_start(synth, phoneme_sentence.c_str(), &options) != PIPER_NCNN_OK) return {};
        
        std::vector<int16_t> pcm_data;
        piper_ncnn_audio_chunk chunk;
        while (piper_ncnn_synthesize_next(synth, &chunk) == PIPER_NCNN_OK) {
            if (chunk.num_samples > 0) {
                pcm_data.insert(pcm_data.end(), chunk.samples, chunk.samples + chunk.num_samples);
            } else {
                break; // End of synthesis
            }
        }
        return pcm_data;
    }
};

// Forward declaration for the BSoundPlayer callback function.
void FillBuffer(void* cookie, void* buffer, size_t size, const media_raw_audio_format& format);

// ============================================================================
// Main Application Class
// ============================================================================

/**
 * @class PiperttsApp
 * @brief The main application class that handles command-line parsing,
 * input processing, and orchestrates the synthesis and playback/output on Haiku.
 */
class PiperttsApp {
public:
    PiperttsApp(int argc, char** argv);
    int run();

private:
    // Member variables to store application state and configuration
    std::vector<std::string> m_args; // Raw command-line arguments
    std::string m_model_dir;         // Full path to the model directory
    int m_speaker_id;                // Speaker ID for multi-speaker models
    std::string m_text_arg;          // Text provided via -t argument
    std::string m_infile;            // Path to input text file
    std::string m_outfile;           // Path to output WAV file
    std::string m_espeak_lang_override; // espeak voice to override config

    std::string m_piper_lang;        // Language of the Piper model (from config)
    std::string m_espeak_voice;      // espeak voice for phonemes (from config)
    
    PiperEngine m_engine;            // The TTS engine instance
    std::queue<std::vector<int16_t>> m_audio_queue; // Buffer for audio data before playback
    std::mutex m_queue_mutex;        // Mutex to protect the audio queue from concurrent access

    // Private helper methods
    void parse_arguments();
    void print_help();
    
    std::string find_model_name_from_dir(const std::string& model_dir);
    void parse_config(const std::string& path);
    std::string escape_for_single_quotes(const std::string& text);
    std::string get_phonemes(const std::string& lang, const std::string& text);
    std::vector<std::string> text_to_phoneme_sentences(const std::string& text);
    std::string normalize_persian_punctuation(const std::string& input);

    // Make the global FillBuffer function a friend to allow it to access private members.
    friend void FillBuffer(void* cookie, void* buffer, size_t size, const media_raw_audio_format& format);
};

// ============================================================================
// Implementation of PiperttsApp methods
// ============================================================================

PiperttsApp::PiperttsApp(int argc, char** argv) : m_speaker_id(0) {
    if (argc > 0) {
        m_args.assign(argv, argv + argc);
    }
}

/**
 * @brief Parses command-line arguments and configures the application member variables.
 */
void PiperttsApp::parse_arguments() {
    // First, check if the user is asking for help.
    for (size_t i = 1; i < m_args.size(); ++i) {
        if (m_args[i] == "-h" || m_args[i] == "--help") {
            print_help();
            exit(0);
        }
    }

    // Default path for models on Haiku.
    std::string model_path = "/boot/home/config/non-packaged/data/pipertts/models/";
    std::string model_name = "";
    bool model_name_provided = false;
    
    // Loop through arguments to find and process known flags.
    for (size_t i = 1; i < m_args.size(); ++i) {
        std::string arg = m_args[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < m_args.size()) {
            model_name = m_args[++i];
            model_name_provided = true;
        } else if ((arg == "-p" || arg == "--path") && i + 1 < m_args.size()) {
            model_path = m_args[++i];
        } else if ((arg == "-s" || arg == "--speaker_id") && i + 1 < m_args.size()) {
            m_speaker_id = std::stoi(m_args[++i]);
        } else if ((arg == "-l" || arg == "--lang") && i + 1 < m_args.size()) {
            m_espeak_lang_override = m_args[++i];
        } else if ((arg == "-t" || arg == "--text") && i + 1 < m_args.size()) {
            m_text_arg = m_args[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < m_args.size()) {
            m_outfile = m_args[++i];
        } else if ((arg == "-f" || arg == "--file") && i + 1 < m_args.size()) {
            m_infile = m_args[++i];
        }
    }

    // The -m flag is required.
    if (!model_name_provided) {
        throw std::runtime_error("Mandatory argument -m (--model) is missing. Use -h for help.");
    }

    // Construct the full path to the model directory.
    if (!model_path.empty() && model_path.back() == '/') {
        m_model_dir = model_path + model_name;
    } else {
        m_model_dir = model_path + "/" + model_name;
    }

    // Find the model's configuration file and parse it.
    std::string model_name_base = find_model_name_from_dir(m_model_dir);
    std::string config_path = m_model_dir + "/" + model_name_base + "_config.txt";
    parse_config(config_path);
}

/**
 * @brief The main execution method of the application.
 * @return 0 on success, non-zero on failure.
 */
int PiperttsApp::run() {
    // 1. Configure the app from command-line arguments.
    parse_arguments();
    // 2. Initialize the Piper TTS engine.
    m_engine.init(m_model_dir, m_piper_lang, m_speaker_id);
    
    std::string final_espeak_voice = m_espeak_lang_override.empty() ? m_espeak_voice : m_espeak_lang_override;

    // 3. Prepare for output: either a WAV file writer or the Haiku BSoundPlayer.
    // Use std::unique_ptr for WavWriter for automatic and safer memory management.
    std::unique_ptr<WavWriter> writer = nullptr; 
    // BSoundPlayer is managed with a raw pointer as is common with Haiku APIs.
    BSoundPlayer* player = nullptr;

    if (!m_outfile.empty()) {
        // If an output file is specified, create a WavWriter.
        writer = std::make_unique<WavWriter>(m_outfile, 22050, 1, 16);
    } else {
        // Otherwise, set up the BSoundPlayer for direct audio playback.
        media_raw_audio_format format;
        memset(&format, 0, sizeof(format));
        format.frame_rate = 22050; 
        format.channel_count = 1; 
        format.format = media_raw_audio_format::B_AUDIO_SHORT;
        format.byte_order = B_MEDIA_LITTLE_ENDIAN; 
        format.buffer_size = 4096;
        // Create a new BSoundPlayer instance, passing `this` app object as the "cookie".
        player = new BSoundPlayer(&format, "pipertts", FillBuffer, NULL, this);
        player->Start();
        player->SetHasData(true);
    }

    // 4. Set up the input text stream (from argument, file, or stdin).
    std::istream* input_stream_ptr = nullptr;
    std::ifstream file_stream;
    std::stringstream text_stream;

    if (!m_text_arg.empty()) {
        text_stream.str(m_text_arg);
        input_stream_ptr = &text_stream;
    } else if (!m_infile.empty()) {
        file_stream.open(m_infile);
        if (!file_stream) throw std::runtime_error("Could not open input file: " + m_infile);
        input_stream_ptr = &file_stream;
    } else {
        input_stream_ptr = &std::cin;
    }

    // 5. Main processing loop: read line by line from input.
    std::string line;
    while (std::getline(*input_stream_ptr, line)) {
        if (line.empty()) continue;
        
        std::string text_to_process = (m_piper_lang == "fa") ? normalize_persian_punctuation(line) : line;
        std::vector<std::string> phonemes_list = text_to_phoneme_sentences(text_to_process);

        for (const auto& p_sentence : phonemes_list) {
            // Synthesize audio from the phoneme sentence.
            auto pcm = m_engine.synthesize(p_sentence);

            // 6. Direct the synthesized audio to the appropriate output.
            if (writer) {
                // If writing to a file, use the writer.
                writer->write(pcm);
            } else if (player && !pcm.empty()) {
                // If playing audio, push the data to the thread-safe queue.
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                m_audio_queue.push(pcm);
            }
        }
    }
    if (file_stream.is_open()) file_stream.close();

    // 7. Cleanup resources.
    if (writer) {
        writer->close(); // Finalize the WAV file.
        // No need to call 'delete writer', std::unique_ptr handles it automatically.
    }
    if (player) {
        // Wait for the audio queue to be fully consumed by the BSoundPlayer callback.
        while (true) {
            bool queue_is_empty;
            { 
                std::lock_guard<std::mutex> lock(m_queue_mutex); 
                queue_is_empty = m_audio_queue.empty(); 
            }
            if (queue_is_empty) { 
                usleep(100000); // Wait a little longer to ensure the last buffer is played.
                break; 
            }
            usleep(50000); // Wait before checking again.
        }
        player->Stop();
        delete player;
    }
    return 0;
}


/**
 * @brief Displays the command-line help message.
 */
void PiperttsApp::print_help() {
    // This help text remains unchanged as requested.
    std::cout << R"(
Usage: pipertts -m <model_name> [OPTIONS]

A command-line Text-to-Speech (TTS) program for the Haiku OS using Piper NCNN models.
The program automatically detects the model configuration from the specified directory.
Input text is read with the following priority: -t > -f > stdin.

## Mandatory Argument:
  -m, --model <model_name>  The name of the model directory to use.

## Options:
  -p, --path <path>         Path to the base directory containing model folders.
                            (Default: /boot/home/config/non-packaged/data/pipertts/models/)

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

## Guidelines for Converting Piper ONNX Model

**References:**
* https://github.com/nihui/ncnn-android-piper
* https://github.com/OHF-Voice/piper1-gpl
* https://huggingface.co/datasets/rhasspy/piper-checkpoints

**Steps to convert Piper checkpoints to NCNN models:**

1.  **Checkout the correct version of the piper repository:**
    ```bash
    git clone [https://github.com/OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)
    cd piper1-gpl
    git checkout 113931937cf235fc8a11afd1ca4be209bc6919bc7
    ```

2.  **Apply the necessary patch:**
    ```bash
    # Ensure 'piper1-gpl.patch' is available
    git apply piper1-gpl.patch
    ```

3.  **Set up the Python environment and install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -e .[train]
    ```

4.  **Download a Piper checkpoint file (`.ckpt`) from Hugging Face:**
    https://huggingface.co/datasets/rhasspy/piper-checkpoints

5.  **Install the PNNX model converter:**
    ```bash
    pip install -U pnnx
    ```

6.  **Obtain the `export_ncnn.py` script.**

7.  **Run the conversion script on your checkpoint file:**
    ```bash
    # Replace with your actual file
    python export_ncnn.py (language code).ckpt (e.g., en.ckpt, fa.ckpt, ...)
    ```
8.  **The usable converted models can be downloaded from the following link:**
    * https://huggingface.co/gyroing/PiperTTS-NCNN-Models/tree/main
--------------------------------------------------------------------------------

**Created by:** gyroing (Amir Hossein Navabi)
* **GitHub:** https://github.com/gyroing
)" << std::endl;
}

/**
 * @brief Finds the base name of a model by looking for a unique `*_config.txt` file in a directory.
 * @param model_dir The directory to search.
 * @return The base name of the model.
 */
std::string PiperttsApp::find_model_name_from_dir(const std::string& model_dir) {
    DIR *dir = opendir(model_dir.c_str());
    if (!dir) throw std::runtime_error("Could not open model directory: " + model_dir);

    std::string model_name = "";
    int count = 0;
    const std::string suffix = "_config.txt";
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        std::string filename = ent->d_name;
        if (filename.length() > suffix.length() && filename.substr(filename.length() - suffix.length()) == suffix) {
            model_name = filename.substr(0, filename.length() - suffix.length());
            count++;
        }
    }
    closedir(dir);

    if (count == 0) throw std::runtime_error("No '_config.txt' file found in " + model_dir);
    if (count > 1) throw std::runtime_error("Multiple '_config.txt' files found. Directory must contain only one model.");
    return model_name;
}

/**
 * @brief Parses the model's config file to get required parameters.
 * @param path The full path to the `_config.txt` file.
 */
void PiperttsApp::parse_config(const std::string& path) {
    std::ifstream config_file(path);
    if (!config_file.is_open()) throw std::runtime_error("Cannot open config file: " + path);
    std::string line;
    while (std::getline(config_file, line)) {
        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, '=') && std::getline(ss, value)) {
            auto trim_string = [](const std::string& s, const std::string& delimiters = " \t\n\r\"") {
                size_t first = s.find_first_not_of(delimiters);
                if (std::string::npos == first) return s;
                size_t last = s.find_last_not_of(delimiters);
                return s.substr(first, (last - first + 1));
            };
            key = trim_string(key);
            value = trim_string(value);
            if (key == "espeak_voice") m_espeak_voice = value; 
            else if (key == "piper_language") m_piper_lang = value;
        }
    }
    if (m_espeak_voice.empty() || m_piper_lang.empty()) {
        throw std::runtime_error("'espeak_voice' or 'piper_language' not found in config file.");
    }
}

/**
 * @brief Escapes single quotes in a string to prevent command injection.
 * @param text The input string.
 * @return The escaped string.
 */
std::string PiperttsApp::escape_for_single_quotes(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    for (char c : text) {
        if (c == '\'') result.append("'\\''");
        else result.push_back(c);
    }
    return result;
}

/**
 * @brief Converts text to IPA phonemes by calling an external `espeak` process.
 * @param lang The espeak voice/language code (e.g., "en-us").
 * @param text The text to convert.
 * @return A string containing the IPA phonemes.
 */
std::string PiperttsApp::get_phonemes(const std::string& lang, const std::string& text) {
    if (text.empty() || text.find_first_not_of(" \t\n\r") == std::string::npos) return "";
    std::string escaped_text = escape_for_single_quotes(text);
    // "2>/dev/null" redirects stderr to null, hiding espeak's messages.
    std::string command = "espeak -q -v " + lang + " --ipa '" + escaped_text + "' 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) result += buffer;
    pclose(pipe);
    if (!result.empty() && result.back() == '\n') result.pop_back(); // Trim trailing newline
    return result;
}

/**
 * @brief Splits a long text into smaller sentences based on punctuation and gets phonemes for each.
 * @param text The full input text.
 * @return A vector of phoneme strings, one for each sentence.
 */
std::vector<std::string> PiperttsApp::text_to_phoneme_sentences(const std::string& text) {
    std::vector<std::string> phoneme_sentences;
    std::string final_espeak_voice = m_espeak_lang_override.empty() ? m_espeak_voice : m_espeak_lang_override;
    const std::string punct = ".,:;!?";
    size_t start_pos = 0;
    while (start_pos < text.length()) {
        size_t end_pos = text.find_first_of(punct, start_pos);
        std::string sentence_chunk;
        char punctuation_char = 0;
        if (end_pos == std::string::npos) {
            sentence_chunk = text.substr(start_pos);
            start_pos = text.length();
        } else {
            sentence_chunk = text.substr(start_pos, end_pos - start_pos);
            punctuation_char = text[end_pos];
            start_pos = end_pos + 1;
        }
        if (!sentence_chunk.empty() || punctuation_char != 0) {
            std::string chunk_phonemes = get_phonemes(final_espeak_voice, sentence_chunk);
            if (!chunk_phonemes.empty()) {
                 if (punctuation_char != 0) {
                    phoneme_sentences.push_back(chunk_phonemes + punctuation_char);
                 } else {
                    phoneme_sentences.push_back(chunk_phonemes);
                 }
            }
        }
    }
    return phoneme_sentences;
}

/**
 * @brief Converts common UTF-8 Persian punctuation to their ASCII equivalents.
 * @param input The input string.
 * @return The normalized string.
 */
std::string PiperttsApp::normalize_persian_punctuation(const std::string& input) {
    std::string output;
    output.reserve(input.length());
    for (size_t i = 0; i < input.length(); ++i) {
        const unsigned char c1 = static_cast<unsigned char>(input[i]);
        const unsigned char c2 = (i + 1 < input.length()) ? static_cast<unsigned char>(input[i+1]) : 0;
        if (c1 == 0xD8 && c2 == 0x9F) { output += '?'; i++; }      // Persian Question Mark
        else if (c1 == 0xD8 && c2 == 0x8C) { output += ','; i++; } // Persian Comma
        else if (c1 == 0xD8 && c2 == 0x9B) { output += ';'; i++; } // Persian Semicolon
        else { output += input[i]; }
    }
    return output;
}

// ============================================================================
// Global Functions
// ============================================================================

/**
 * @brief The callback function for Haiku's BSoundPlayer.
 * This function is called by the media server in a separate thread whenever it needs more audio data.
 * @param cookie A pointer to the PiperttsApp instance.
 * @param buffer A pointer to the audio buffer to be filled.
 * @param size The size of the buffer in bytes.
 * @param format The requested audio format.
 */
void FillBuffer(void* cookie, void* buffer, size_t size, const media_raw_audio_format& format) {
    PiperttsApp* app = static_cast<PiperttsApp*>(cookie);
    if (!app) return;

    // Lock the mutex to safely access the shared audio queue.
    std::lock_guard<std::mutex> lock(app->m_queue_mutex);
    size_t bytes_written = 0;
    // Keep filling the buffer as long as there is space and data is available in the queue.
    while (bytes_written < size && !app->m_audio_queue.empty()) {
        auto& chunk = app->m_audio_queue.front();
        size_t chunk_bytes = chunk.size() * sizeof(int16_t);
        size_t bytes_to_copy = std::min(chunk_bytes, size - bytes_written);
        
        memcpy((char*)buffer + bytes_written, chunk.data(), bytes_to_copy);
        bytes_written += bytes_to_copy;

        if (bytes_to_copy == chunk_bytes) {
            // If the entire chunk was copied, remove it from the queue.
            app->m_audio_queue.pop();
        } else {
            // If only part of the chunk was copied, resize the chunk to remove the copied part.
            auto& front_chunk = app->m_audio_queue.front();
            front_chunk.erase(front_chunk.begin(), front_chunk.begin() + (bytes_to_copy / sizeof(int16_t)));
        }
    }
    // If the queue becomes empty before the buffer is full, fill the rest with silence (zeros).
    if (bytes_written < size) {
        memset((char*)buffer + bytes_written, 0, size - bytes_written);
    }
}

// ============================================================================
// Main function - Program Entry Point
// ============================================================================
int main(int argc, char** argv) {
    // Use a try-catch block for robust error handling.
    try {
        PiperttsApp app(argc, argv);
        return app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
