include(ExternalProject)

set(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/gtest)
set(GTEST_INCLUDE_DIRS ${THIRD_PARTY_PATH}/gtest/include)
set(GTEST_LIBRARIES ${THIRD_PARTY_PATH}/gtest/lib/libgtest.a
                    ${THIRD_PARTY_PATH}/gtest/lib/libgtest_main.a)

ExternalProject_add(
    extern_gtest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.8.0
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
    GIT_SHALLOW
)

add_library(gtest STATIC IMPORTED GLOBAL)
set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARIES})
add_dependencies(gtest extern_gtest)

include_directories(${THIRD_PARTY_PATH}/gtest/include)