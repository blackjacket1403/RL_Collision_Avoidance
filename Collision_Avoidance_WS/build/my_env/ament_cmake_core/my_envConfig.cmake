# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_my_env_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED my_env_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(my_env_FOUND FALSE)
  elseif(NOT my_env_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(my_env_FOUND FALSE)
  endif()
  return()
endif()
set(_my_env_CONFIG_INCLUDED TRUE)

# output package information
if(NOT my_env_FIND_QUIETLY)
  message(STATUS "Found my_env: 0.0.0 (${my_env_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'my_env' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${my_env_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(my_env_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${my_env_DIR}/${_extra}")
endforeach()
