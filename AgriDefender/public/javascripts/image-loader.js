function imagenetClassesTopK(classProbabilities, k) {
  if (!k) { k = 5; }
  const probs = Array.from(classProbabilities);
  const probsIndices = probs.map(
    function (prob, index) {
      return [prob, index];
    }
  );
  const sorted = probsIndices.sort(
    function (a, b) {
      if (a[0] < b[0]) {
        return -1;
      }
      if (a[0] > b[0]) {
        return 1;
      }
      return 0;
    }
  ).reverse();
  const topK = sorted.slice(0, k).map(function (probIndex) {
      const iClass = $.index.var.classindex[probIndex[1]];
      return {
          id: iClass,
          index: parseInt(probIndex[1], 10),
          name: iClass.replace(/_/g, ' '),
          probability: probIndex[0] * 100
      };
  });
  return topK;
}
function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);

  //Normalize 0-255 to (-1)-1
  // ndarray.ops.divseq(dataFromImage, 255.0);
  
  // ndarray.ops.subseq(dataFromImage.pick(null, null, 2), 0.4914);
  // ndarray.ops.divseq(dataFromImage.pick(null, null, 2), 0.2471);
  // ndarray.ops.subseq(dataFromImage.pick(null, null, 1), 0.4822);
  // ndarray.ops.divseq(dataFromImage.pick(null, null, 1), 0.2435);
  // ndarray.ops.subseq(dataFromImage.pick(null, null, 0), 0.4465);
  // ndarray.ops.divseq(dataFromImage.pick(null, null, 0), 0.2616);

  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));
  return dataProcessed.data;
}