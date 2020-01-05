# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindMKL
# -------
#
# Find a Intel® Math Kernel Library (Intel® MKL) installation and provide
# all necessary variables and macros to compile software for it.
#
# MKLROOT is required in your system
#
# we use mkl_link_tool to get the library needed depending on variables
# There are few sets of libraries:
#
# Array indexes modes:
#
# ::
#
# LP - 32 bit indexes of arrays
# ILP - 64 bit indexes of arrays
#
#
#
# Threading:
#
# ::
#
# SEQUENTIAL - no threading
# INTEL - Intel threading library
# GNU - GNU threading library
# MPI support
# NOMPI - no MPI support
# INTEL - Intel MPI library
# OPEN - Open MPI library
# SGI - SGI MPT Library
#
#
#
#
# The following are set after the configuration is done:
#
# ::
#
#  MKL_FOUND        -  system has MKL
#  MKL_ROOT_DIR     -  path to the MKL base directory
#  MKL_INCLUDE_DIR  -  the MKL include directory
#  MKL_LIBRARIES    -  MKL libraries
#  MKL_LIBRARY_DIR  -  MKL library dir (for dlls!)
#
#
#
# Sample usage:
#
# If MKL is required (i.e., not an optional part):
#
# ::
#
#    find_package(MKL REQUIRED)
#    if (MKL_FOUND)
#        include_directories(${MKL_INCLUDE_DIR})
#        # and for each of your dependent executable/library targets:
#        target_link_libraries(<YourTarget> ${MKL_LIBRARIES})
#    endif()


# NOTES
#
# If you want to use the module and your build type is not supported
# out-of-the-box, please contact me to exchange information on how
# your system is setup and I'll try to add support for it.
#
# AUTHOR
#
# Joan MASSICH (joan.massich-vall.AT.inria.fr).
# Alexandre GRAMFORT (Alexandre.Gramfort.AT.inria.fr)
# Théodore PAPADOPOULO (papadop.AT.inria.fr)


set(CMAKE_FIND_DEBUG_MODE 1)

# unset this variable defined in matio
unset(MSVC)

# Find MKL ROOT
find_path(MKL_ROOT_DIR NAMES include/mkl_cblas.h PATHS $ENV{MKLROOT})

# Convert symlinks to real paths

get_filename_component(MKL_ROOT_DIR ${MKL_ROOT_DIR} REALPATH)

if (NOT MKL_ROOT_DIR AND NOT TARGET mkl::mkl)
    if (MKL_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find MKL: please set environment variable {MKLROOT}")
    else()
        unset(MKL_ROOT_DIR CACHE)
    endif()
else()
    set(MKL_INCLUDE_DIR ${MKL_ROOT_DIR}/include)
#
#    # set arguments to call the mkl provided tool for linking
#    set(mkl_link_tool ${mkl_root_dir}/tools/mkl_link_tool)
#
#    if (win32)
#        set(mkl_link_tool ${mkl_link_tool}.exe)
#    endif()
#
#    # check that the tools exists or quit
#    if (not exists "${mkl_link_tool}")
#        message(fatal_error "cannot find mkl tool: ${mkl_link_tool}")
#    endif()
#
#    # first the libs
#    set(mkl_link_tool_command ${mkl_link_tool} "-libs")
#
#    if (cmake_cxx_compiler_id strequal "clang")
#        list(append mkl_link_tool_command "--compiler=clang")
#    elseif(cmake_cxx_compiler_id strequal "intel")
#        list(append mkl_link_tool_command "--compiler=intel_c")
#    elseif(cmake_cxx_compiler_id strequal "msvc")
#        list(append mkl_link_tool_command "--compiler=ms_c")
#    else()
#        list(append mkl_link_tool_command "--compiler=gnu_c")
#    endif()
#
#    if (apple)
#        list(append mkl_link_tool_command "--os=mac")
#    elseif(win32)
#        list(append mkl_link_tool_command "--os=win")
#    else()
#        list(append mkl_link_tool_command "--os=lnx")
#    endif()
#
#    set(mkl_lib_dir)
#    if (${cmake_sizeof_void_p} equal 8)
#        list(append mkl_link_tool_command "--arch=intel64")
#        set(mkl_lib_dir "intel64")
#    else()
#        list(append mkl_link_tool_command "--arch=ia32")
#        set(mkl_lib_dir "ia32")
#    endif()
#
#    if (mkl_use_sdl)
#        list(append mkl_link_tool_command "--linking=sdl")
#    else()
#        if (bla_static)
#            list(append mkl_link_tool_command "--linking=static")
#        else()
#            list(append mkl_link_tool_command "--linking=dynamic")
#        endif()
#    endif()
#
#    if (mkl_use_parallel)
#        list(append mkl_link_tool_command "--parallel=yes")
#    else()
#        list(append mkl_link_tool_command "--parallel=no")
#    endif()
#
#    if (force_build_32bits)
#        list(append mkl_link_tool_command "--interface=cdecl")
#        set(mkl_use_interface "cdecl" cache string "disabled by force_build_32bits" force)
#    else()
#        list(append mkl_link_tool_command "--interface=${mkl_use_interface}")
#    endif()
#
#    if (mkl_use_parallel)
#        if (unix and not apple)
#            list(append mkl_link_tool_command "--openmp=gomp")
#        else()
#            list(append mkl_link_tool_command "--threading-library=iomp5")
#            list(append mkl_link_tool_command "--openmp=iomp5")
#        endif()
#    endif()
#
#    execute_process(command ${mkl_link_tool_command}
#            output_variable mkl_libs
#            result_variable command_worked
#            timeout 2 error_quiet)
#
#    set(mkl_libraries)
#
#    if (not ${command_worked} equal 0)
#        message(fatal_error "cannot find the mkl libraries correctly. please check your mkl input variables and mkl_link_tool. the command executed was:\n ${mkl_link_tool_command}.")
#    endif()
#
#    set(mkl_library_dir)
#
#    if (win32)
#        set(mkl_library_dir "${mkl_root_dir}/lib/${mkl_lib_dir}/" "${mkl_root_dir}/../compiler/lib/${mkl_lib_dir}")
#
#        # remove unwanted break
#        string(regex replace "\n" "" mkl_libs ${mkl_libs})
#
#        # get the list of libs
#        separate_arguments(mkl_libs)
#        foreach(i ${mkl_libs})
#            find_library(fullpath_lib ${i} paths "${mkl_library_dir}")
#
#            if (fullpath_lib)
#                list(append mkl_libraries ${fullpath_lib})
#            elseif(i)
#                list(append mkl_libraries ${i})
#            endif()
#            unset(fullpath_lib cache)
#        endforeach()
#
#    else() # unix and macos
#        # remove unwanted break
#        string(regex replace "\n" "" mkl_libs ${mkl_libs})
#        if (mkl_link_tool_command matches "static")
#            string(replace "$(mklroot)" "${mkl_root_dir}" mkl_libraries ${mkl_libs})
#            # hack for lin with libiomp5.a
#            if (apple)
#                string(replace "-liomp5" "${mkl_root_dir}/../compiler/lib/libiomp5.a" mkl_libraries ${mkl_libraries})
#            else()
#                string(replace "-liomp5" "${mkl_root_dir}/../compiler/lib/${mkl_lib_dir}/libiomp5.a" mkl_libraries ${mkl_libraries})
#            endif()
#            separate_arguments(mkl_libraries)
#        else() # dynamic or sdl
#            # get the lib dirs
#            string(regex replace "^.*-l[^/]+([^\ ]+).*" "${mkl_root_dir}\\1" intel_lib_dir ${mkl_libs})
#            if (not exists ${intel_lib_dir})
#                #   work around a bug in mkl 2018
#                set(intel_lib_dir1 "${intel_lib_dir}_lin")
#                if (not exists ${intel_lib_dir1})
#                    message(fatal_error "mkl installation broken. directory ${intel_lib_dir} does not exist.")
#                endif()
#                set(intel_lib_dir ${intel_lib_dir1})
#            endif()
#            set(mkl_library_dir ${intel_lib_dir} "${mkl_root_dir}/../compiler/lib/${mkl_lib_dir}")
#
#            # get the list of libs
#            separate_arguments(mkl_libs)
#
#            # set full path to libs
#            foreach(i ${mkl_libs})
#                string(regex replace " -" "-" i ${i})
#                string(regex replace "-l([^\ ]+)" "\\1" i ${i})
#                string(regex replace "-l.*" "" i ${i})
#
#                find_library(fullpath_lib ${i} paths "${mkl_library_dir}")
#
#                if (fullpath_lib)
#                    list(append mkl_libraries ${fullpath_lib})
#                elseif(i)
#                    list(append mkl_libraries ${i})
#                endif()
#                unset(fullpath_lib cache)
#            endforeach()
#        endif()
#    endif()
#
#    # now definitions
#    string(replace "-libs" "-opts" mkl_link_tool_command "${mkl_link_tool_command}")
#    execute_process(command ${mkl_link_tool_command} output_variable result_opts timeout 2 error_quiet)
#    string(regex matchall "[-/]d[^\ ]*" mkl_definitions ${result_opts})
#
#    if (cmake_find_debug_mode)
#        message(status "exectuted command: \n${mkl_link_tool_command}")
#        message(status "found mkl_libraries:\n${mkl_libraries} ")
#        message(status "found mkl_definitions:\n${mkl_definitions} ")
#        message(status "found mkl_library_dir:\n${mkl_library_dir} ")
#        message(status "found mkl_include_dir:\n${mkl_include_dir} ")
#    endif()
#
#    include(findpackagehandlestandardargs)
#    find_package_handle_standard_args(mkl default_msg mkl_include_dir mkl_libraries)
#
#    mark_as_advanced(mkl_include_dir mkl_libraries mkl_definitions mkl_root_dir)

    add_library(mkl::mkl SHARED IMPORTED)
    set_target_properties(mkl::mkl PROPERTIES
            IMPORTED_LOCATION "${MKL_ROOT_DIR}/lib/intel64/libmkl_rt.so"
            INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
    target_link_options(mkl::mkl BEFORE INTERFACE "-L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed")
    find_package(Threads REQUIRED)
    target_link_libraries(mkl::mkl INTERFACE Threads::Threads m dl)
    #set_property(TARGET mkl::mkl PROPERTY INTERFACE_LINK_OPTIONS "-lm -ldl")

endif()