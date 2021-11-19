aux_source_directory(. PLAYGROUND)

foreach (file ${PLAYGROUND})
  string(REPLACE "./" "" filename ${file})
  string(REPLACE ".cc" "" appname ${filename})
  add_executable(${appname} ${filename})
endforeach()