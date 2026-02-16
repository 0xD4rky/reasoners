#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "load_tokenizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qwen_tokenizer_cpp, m) {
    m.doc() = "Fast C++ tokenizer for Qwen 2.5 models";

    py::class_<QwenBPETokenizer::Config>(m, "TokenizerConfig")
        .def(py::init<>())
        .def_readwrite("add_prefix_space", &QwenBPETokenizer::Config::add_prefix_space)
        .def_readwrite("trim_offsets", &QwenBPETokenizer::Config::trim_offsets);

    py::class_<QwenBPETokenizer>(m, "QwenBPETokenizer")
        .def(py::init<const std::string&>(), py::arg("tokenizer_json_path"))
        .def("encode", &QwenBPETokenizer::encode,
             py::arg("text"),
             py::arg("allowed_special") = std::unordered_set<std::string>())
        .def("batch_encode", &QwenBPETokenizer::batch_encode,
             py::arg("texts"),
             py::arg("allowed_special") = std::unordered_set<std::string>())
        .def("decode", &QwenBPETokenizer::decode,
             py::arg("ids"))
        .def("batch_decode", &QwenBPETokenizer::batch_decode,
             py::arg("batch_ids"))
        .def("token_to_id", &QwenBPETokenizer::token_to_id,
             py::arg("token"))
        .def("config", &QwenBPETokenizer::config);
}
