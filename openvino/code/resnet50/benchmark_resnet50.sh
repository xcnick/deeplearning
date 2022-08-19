#!/bin/bash

echo "OpenVINO Resnet50 Benchmark"

hints="latency throughput"
apis="sync async"
batchs="1 2 4 8 16"

for hint in $hints;
do
  for api in $apis;
  do
    for batch in $batchs;
    do
      echo "Running benchmark for resnet50 $hint $api $batch"
      benchmark_app -m resnet50.onnx -hint $hint -api $api -b $batch > logs/resnet50/fp32/$hint-$api-$batch.log
    done
  done
done
