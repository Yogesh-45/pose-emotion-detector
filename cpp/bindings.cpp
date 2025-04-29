#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

PYBIND11_MODULE(emotion_binding, m) {
    m.def("detect_emotion", [](const std::string& path) {
        // Add the directory containing your Python script to sys.path
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "./utils");

        py::object emotion_api = py::module::import("emotion_api");
        py::object detect_emotion = emotion_api.attr("detect_emotion");
        py::object result = detect_emotion(path.c_str());  // Call detect_emotion with the path

        return result.cast<std::string>();
    }, "Detect emotion from image path");
}
