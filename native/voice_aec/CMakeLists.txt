cmake_minimum_required(VERSION 3.20)
project(tldw_voice_aec LANGUAGES C CXX)

set(WEBRTC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/webrtc")
set(WEBRTC_MANIFEST "${WEBRTC_ROOT}/FILES.sha256")
if(
  NOT EXISTS "${WEBRTC_MANIFEST}"
  OR IS_DIRECTORY "${WEBRTC_MANIFEST}"
  OR IS_SYMLINK "${WEBRTC_MANIFEST}"
)
  message(FATAL_ERROR "Missing pinned WebRTC source manifest: ${WEBRTC_MANIFEST}")
endif()
file(REAL_PATH "${WEBRTC_ROOT}" WEBRTC_ROOT_REAL)

file(STRINGS "${WEBRTC_MANIFEST}" WEBRTC_MANIFEST_LINES)
set(WEBRTC_MANIFEST_PATHS)
set(WEBRTC_SOURCES)
foreach(MANIFEST_LINE IN LISTS WEBRTC_MANIFEST_LINES)
  string(LENGTH "${MANIFEST_LINE}" MANIFEST_LINE_LENGTH)
  if(MANIFEST_LINE_LENGTH LESS 67)
    message(FATAL_ERROR "Malformed WebRTC manifest line: ${MANIFEST_LINE}")
  endif()
  string(SUBSTRING "${MANIFEST_LINE}" 0 64 MANIFEST_HASH)
  string(SUBSTRING "${MANIFEST_LINE}" 64 2 MANIFEST_DELIMITER)
  if(NOT MANIFEST_HASH MATCHES "^[0-9a-f]+$" OR NOT MANIFEST_DELIMITER STREQUAL "  ")
    message(FATAL_ERROR "Malformed WebRTC manifest line: ${MANIFEST_LINE}")
  endif()
  string(SUBSTRING "${MANIFEST_LINE}" 66 -1 RELATIVE_PATH)
  if(
    RELATIVE_PATH STREQUAL ""
    OR RELATIVE_PATH MATCHES "^/"
    OR RELATIVE_PATH MATCHES "^[A-Za-z]:/"
    OR RELATIVE_PATH MATCHES "\\\\"
    OR RELATIVE_PATH MATCHES "(^|/)\\.\\.?($|/)"
    OR RELATIVE_PATH MATCHES "//"
    OR RELATIVE_PATH MATCHES "/$"
  )
    message(
      FATAL_ERROR
      "WebRTC manifest path is not normalized relative POSIX: ${RELATIVE_PATH}"
    )
  endif()
  cmake_path(NORMAL_PATH RELATIVE_PATH OUTPUT_VARIABLE NORMALIZED_PATH)
  if(NOT RELATIVE_PATH STREQUAL NORMALIZED_PATH)
    message(
      FATAL_ERROR
      "WebRTC manifest path is not normalized relative POSIX: ${RELATIVE_PATH}"
    )
  endif()
  set(VENDORED_FILE "${WEBRTC_ROOT}/${RELATIVE_PATH}")
  if(
    NOT EXISTS "${VENDORED_FILE}"
    OR IS_DIRECTORY "${VENDORED_FILE}"
    OR IS_SYMLINK "${VENDORED_FILE}"
  )
    message(FATAL_ERROR "WebRTC manifest file is not a regular file: ${RELATIVE_PATH}")
  endif()
  file(REAL_PATH "${VENDORED_FILE}" VENDORED_FILE_REAL)
  cmake_path(
    IS_PREFIX WEBRTC_ROOT_REAL "${VENDORED_FILE_REAL}" NORMALIZE VENDORED_FILE_UNDER_ROOT
  )
  if(NOT VENDORED_FILE_UNDER_ROOT)
    message(FATAL_ERROR "WebRTC manifest path escapes vendor root: ${RELATIVE_PATH}")
  endif()
  file(SHA256 "${VENDORED_FILE}" VENDORED_FILE_HASH)
  if(NOT VENDORED_FILE_HASH STREQUAL MANIFEST_HASH)
    message(FATAL_ERROR "WebRTC manifest SHA-256 mismatch for ${RELATIVE_PATH}")
  endif()
  list(APPEND WEBRTC_MANIFEST_PATHS "${RELATIVE_PATH}")
  if(RELATIVE_PATH MATCHES "\\.(cc|c|S)$")
    list(APPEND WEBRTC_SOURCES "${VENDORED_FILE}")
  endif()
endforeach()

file(
  GLOB_RECURSE WEBRTC_TREE_PATHS
  LIST_DIRECTORIES true
  RELATIVE "${WEBRTC_ROOT}"
  "${WEBRTC_ROOT}/*"
)
set(WEBRTC_ACTUAL_FILE_PATHS)
foreach(TREE_PATH IN LISTS WEBRTC_TREE_PATHS)
  set(ABSOLUTE_TREE_PATH "${WEBRTC_ROOT}/${TREE_PATH}")
  if(IS_SYMLINK "${ABSOLUTE_TREE_PATH}")
    message(FATAL_ERROR "WebRTC vendor tree contains a symlink: ${TREE_PATH}")
  endif()
  if(NOT IS_DIRECTORY "${ABSOLUTE_TREE_PATH}" AND NOT TREE_PATH STREQUAL "FILES.sha256")
    list(APPEND WEBRTC_ACTUAL_FILE_PATHS "${TREE_PATH}")
  endif()
endforeach()
list(SORT WEBRTC_MANIFEST_PATHS)
list(SORT WEBRTC_ACTUAL_FILE_PATHS)

set(WEBRTC_UNMANIFESTED_PATHS ${WEBRTC_ACTUAL_FILE_PATHS})
foreach(MANIFEST_PATH IN LISTS WEBRTC_MANIFEST_PATHS)
  list(REMOVE_ITEM WEBRTC_UNMANIFESTED_PATHS "${MANIFEST_PATH}")
endforeach()
if(WEBRTC_UNMANIFESTED_PATHS)
  list(JOIN WEBRTC_UNMANIFESTED_PATHS ", " WEBRTC_UNMANIFESTED_TEXT)
  message(FATAL_ERROR "WebRTC vendor tree has unmanifested files: ${WEBRTC_UNMANIFESTED_TEXT}")
endif()

set(WEBRTC_MISSING_PATHS ${WEBRTC_MANIFEST_PATHS})
foreach(ACTUAL_PATH IN LISTS WEBRTC_ACTUAL_FILE_PATHS)
  list(REMOVE_ITEM WEBRTC_MISSING_PATHS "${ACTUAL_PATH}")
endforeach()
if(WEBRTC_MISSING_PATHS)
  list(JOIN WEBRTC_MISSING_PATHS ", " WEBRTC_MISSING_TEXT)
  message(FATAL_ERROR "WebRTC manifest lists missing files: ${WEBRTC_MISSING_TEXT}")
endif()

# The checked-in closure contains mutually exclusive GN platform branches.
# Select exactly the branch for this build; never glob additional source files.
list(FILTER WEBRTC_SOURCES EXCLUDE REGEX "_mips\\.(cc|c)$")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
  list(FILTER WEBRTC_SOURCES EXCLUDE REGEX "common_audio/resampler/sinc_resampler_sse\\.cc$")
  list(FILTER WEBRTC_SOURCES EXCLUDE REGEX "ooura_fft_sse2\\.cc$")
  list(FILTER WEBRTC_SOURCES EXCLUDE REGEX "common_audio/.+_armv7\\.S$")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
  list(FILTER WEBRTC_SOURCES EXCLUDE REGEX "ooura_fft_neon\\.cc$")
  list(FILTER WEBRTC_SOURCES EXCLUDE REGEX "common_audio/.+\\.S$")
else()
  message(FATAL_ERROR "Unsupported WebRTC AEC3 build architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT WEBRTC_SOURCES)
  message(FATAL_ERROR "Pinned WebRTC manifest produced an empty translation-unit list")
endif()

include("${WEBRTC_ROOT}/COMPILER_DEFINES.cmake")
if(APPLE)
  set(TLDW_WEBRTC_PLATFORM macos)
elseif(WIN32)
  set(TLDW_WEBRTC_PLATFORM windows)
elseif(UNIX)
  set(TLDW_WEBRTC_PLATFORM linux)
else()
  message(FATAL_ERROR "Unsupported WebRTC AEC3 operating system")
endif()
tldw_webrtc_defines_for_platform(
  "${TLDW_WEBRTC_PLATFORM}" WEBRTC_COMPILE_DEFINES
)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(webrtc_aec3 STATIC ${WEBRTC_SOURCES})
set_target_properties(webrtc_aec3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  webrtc_aec3
  PUBLIC
    "${WEBRTC_ROOT}"
    "${WEBRTC_ROOT}/third_party/abseil-cpp"
)
target_compile_features(webrtc_aec3 PUBLIC cxx_std_17)
target_compile_definitions(webrtc_aec3 PUBLIC ${WEBRTC_COMPILE_DEFINES})

if(WIN32)
  target_link_libraries(webrtc_aec3 PUBLIC winmm)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$" AND NOT MSVC)
  target_compile_options(webrtc_aec3 PRIVATE -msse2)
endif()

pybind11_add_module(_native MODULE src/bindings.cpp src/duplex_bridge.cpp src/duplex_bindings.cpp)
target_link_libraries(_native PRIVATE webrtc_aec3)
target_compile_features(_native PRIVATE cxx_std_17)

install(TARGETS _native LIBRARY DESTINATION tldw_voice_aec)
install(
  FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/PYBIND11_LICENSE.txt"
    "${WEBRTC_ROOT}/COMPILER_DEFINES.cmake"
    "${WEBRTC_ROOT}/UPSTREAM.json"
    "${WEBRTC_ROOT}/FILES.sha256"
    "${WEBRTC_ROOT}/PRISTINE_FILES.sha256"
    "${WEBRTC_ROOT}/OOURA_LICENSE"
    "${WEBRTC_ROOT}/PATCHES.md"
    "${WEBRTC_ROOT}/LICENSE"
    "${WEBRTC_ROOT}/PATENTS"
  DESTINATION tldw_voice_aec/provenance
)
