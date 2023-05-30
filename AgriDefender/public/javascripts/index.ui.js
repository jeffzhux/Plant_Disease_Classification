(function (window, document, $, undefined) {
    $.extend({ index: {} });
    $.extend($.index, { var: {}, ui: {} });
    $.extend($.index.var, {
        classindex:['K0', 'Mg0', 'OT01', 'gDE03', 'gDP04', 'gDP06', 'tDA12', 'tDC01', 'tDC08', 'tDE03', 'tDP06', 'tDS07', 'tID03', 'tIH04', 'tIH05', 'tIL11', 'tIL13'],
        sess : null,
        loadingModelPromise : null
    });
    $.extend($.index.ui, {
        _init: function(){
            $("#dropArea").on('click', function(){
                $('#image').click();
            });
            $("#dropArea").on('drop', function(event){
                event.preventDefault();
                
                const file = event.originalEvent.dataTransfer.files[0];
                if (file) {
                    $.index.ui.handleFile(file);
                }
            });
            $("#dropArea").on('dragover', function(event){
                event.preventDefault();
            });
            $("#image").on('change', function(){
                const file = this.files[0];
                if (file) {
                    $.index.ui.handleFile(file);
                    
                }
            });
        },
        handleFile: function(file){
            // Display a loading message while classification is in progress
            $('#result').hide();
            $('#classification').text('Classifying image...');
            $('#result').show();

            const reader = new FileReader();
            reader.onloadend = async function(event) {
                var img = new Image();
                
                img.onload = function(){
                    var canvas = document.getElementById('canvas');
                    const ctx = canvas.getContext('2d');
                    var maxWidth = canvas.width;
                    var maxHeight = canvas.height;
                    var width = img.width;
                    var height = img.height;
                    if (width > maxWidth || height > maxHeight) {
                        var ratio = Math.min(maxWidth / width, maxHeight / height);
                        width *= ratio;
                        height *= ratio;
                    }
                    ctx.clearRect(0, 0, maxWidth, maxHeight);
                    var x = (canvas.width - width) / 2;
                    var y = (canvas.height - height) / 2;
                    ctx.drawImage(img, x, y, width, height);
                    $('#imagePreview').show();
                    recongnize();
                }
                img.id = "test";
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);

        
        },
        

    })
    async function recongnize(){
        var canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        preprocessData = preprocess(imgData.data, canvas.width, canvas.height)

        const input = new onnx.Tensor(preprocessData, "float32", [1, 3,canvas.height, canvas.width]);
        console.log(input)
        
        const outputMap = await $.index.var.sess.run([input]);
        const outputTensor = outputMap.values().next().value.data;

        topk = imagenetClassesTopK(outputTensor)
        const resultText = topk.map(p => `${p.name}: ${p.probability.toFixed(4)}`).join('<br>');
          $('#classification').html(resultText);
    }
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
      
        // Normalize 0-255 to (-1)-1
        ndarray.ops.divseq(dataFromImage, 255.0);
        ndarray.ops.subseq(dataFromImage, 0.5);
        ndarray.ops.divseq(dataFromImage, 0.5);

        // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
        ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
        ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
        ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));
      
        return dataProcessed.data;
    }
    $(window).ready(function(){
        $.index.var.sess = new onnx.InferenceSession(),
        $.index.var.sess.loadModel("../weights/two_expert.onnx").then(() =>{
            $.index.ui._init();
            $('#dropArea').html("<p>Drag and drop an image here<br/>or click to select an image</p>");
        })
    });

}(window, document, jQuery))