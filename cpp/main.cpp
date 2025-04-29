#include <iostream>
#include <pybind11/embed.h>
#include <Windows.h>  // Required for SetEnvironmentVariable

namespace py = pybind11;

int main() {
    // Set PYTHONHOME
    if (SetEnvironmentVariableA("PYTHONHOME", "C:\\Users\\ShivajiWankhede\\anaconda3\\envs\\mp") == 0) {
        std::cerr << "Failed to set PYTHONHOME." << std::endl;
        return 1;
    }

    // Set PYTHONPATH
    if (SetEnvironmentVariableA("PYTHONPATH", "C:\\Users\\ShivajiWankhede\\anaconda3\\envs\\mp\\Lib;C:\\Users\\ShivajiWankhede\\anaconda3\\envs\\mp\\DLLs") == 0) {
        std::cerr << "Failed to set PYTHONPATH." << std::endl;
        return 1;
    }

    // Set PYTHONIOENCODING
    if (SetEnvironmentVariableA("PYTHONIOENCODING", "UTF-8") == 0) {
        std::cerr << "Failed to set PYTHONIOENCODING." << std::endl;
        return 1;
    }

    // Set PYTHONUTF8
    if (SetEnvironmentVariableA("PYTHONUTF8", "1") == 0) {
        std::cerr << "Failed to set PYTHONUTF8." << std::endl;
        return 1;
    }

    std::cout << ">> Starting Python interpreter" << std::endl;

    try {
        py::scoped_interpreter guard{};  // Start Python interpreter
        std::cout << ">> Python interpreter started successfully" << std::endl;

        // Add the directory containing your Python script to sys.path
        std::cout << ">> Getting sys module" << std::endl;
        py::module sys = py::module::import("sys");
        std::cout << ">> Sys module gotten" << std::endl;

        std::cout << ">> Getting sys.path" << std::endl;
        py::object path = sys.attr("path");
        std::cout << ">> Sys.path gotten" << std::endl;

        std::cout << ">> Inserting path" << std::endl;
        path.attr("insert")(0, "./utils");
        std::cout << ">> Path inserted" << std::endl;

        std::cout << ">> Importing emotion_api module" << std::endl;
        py::module emotion_api = py::module::import("emotion_api");
        std::cout << ">> emotion_api module imported successfully" << std::endl;

        std::cout << ">> Getting detect_emotion function" << std::endl;
        py::object detect_emotion = emotion_api.attr("detect_emotion");
        std::cout << ">> detect_emotion function retrieved successfully" << std::endl;

        std::string image_path = "./test_data/angry_face.jpg";
        std::cout << ">> Image path: " << image_path << std::endl;

        std::cout << ">> Calling detect_emotion function" << std::endl;
        py::object result = detect_emotion(image_path.c_str());
        std::cout << ">> detect_emotion function called successfully" << std::endl;

        std::cout << "Predicted Emotion: " << result.cast<std::string>() << std::endl;

    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        return 1; // Return a non-zero value to indicate failure
    } catch (const std::exception& e) {
        std::cerr << "C++ error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << ">> Program completed successfully" << std::endl;
    return 0;
}


