(function (window, document, $, undefined) {
    $.extend({ index: {} });
    $.extend($.index, { var: {}, ui: {} });
    $.extend($.index.var, {
        classindex:['K0', 'Mg0', 'OT01', 'gDE03', 'gDP04', 'gDP06', 'tDA12', 'tDC01', 'tDC08', 'tDE03', 'tDP06', 'tDS07', 'tID03', 'tIH04', 'tIH05', 'tIL11', 'tIL13'],
        nameindex:['鉀缺乏', '鎂缺乏', '日燒', '白粉病', '銹病', '露菌病', '藻類', '赤葉枯病','藻斑病','茶餅病','輪班病','煤煙病','潛葉蠅','茶角盲椿象','小桔蚜','茶細蛾','茶姬捲葉蛾'],
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
            $("#plant_image").on('change', function(event){
                const url = $(this).find(":selected").val();
                if(url != ""){
                    fetch(url).then(response => response.blob())
                    .then(file => {
                        $.index.ui.handleFile(file);
                    })
                }
                else{
                    // 清空canvas
                    location.reload();
                }

            });
            $("#image").on('change', function(){
                const file = this.files[0];
                console.log(file);
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
                    // var canvas = document.createElement('canvas');

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
                    ctx.fillStyle ='#717171';
                    ctx.fillRect(0,0,maxWidth,maxHeight);
                    ctx.drawImage(img, x, y, width, height);
                    $("#dropArea").html(canvas);
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
        const start = Date.now()
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        preprocessData = preprocess(imgData.data, canvas.width, canvas.height)

        const input = new onnx.Tensor(preprocessData, "float32", [1, 3,canvas.height, canvas.width]);
        
        const outputMap = await $.index.var.sess.run([input]);
        const outputTensor = outputMap.values().next().value.data;

        topk = imagenetClassesTopK(outputTensor)
        const resultText = topk.map(p => `${p.name}: ${p.probability.toFixed(4)}`).join('<br>');
        
        // show result
        $('#result').hide();
        $('#classification').text('');
        $('#inference_time').text((Date.now()-start)/1000 + 's')
        $('#result').show();

        $('#result .output-class').each(function(key, value){
            $(this).find('.output-label').text(topk[key].name)
            if(key === 0){
                $(this).find('.output-bar').css({"width":topk[key].probability + "px","background":"rgba(42, 106, 150, 0.55)","transition":"width 0.2s ease-in-out 0.2s"})
            }
            else{
                $(this).find('.output-bar').css({"width":topk[key].probability + "px","background":"rgba(42, 106, 150, 0.2)","transition":"width 0.2s ease-in-out 0.2s"})
            }
            $(this).find('.output-value').text(topk[key].probability.toFixed(0)+'%')
        })
    }
    $(window).ready(function(){
        $.index.var.sess = new onnx.InferenceSession(),
        $.index.var.sess.loadModel("../weights/two_expert.onnx").then(() =>{
            $.index.ui._init();
            //$('#dropArea').html("<p>Drag and drop an image here<br/>or click to select an image</p>");
        })
    });

}(window, document, jQuery))