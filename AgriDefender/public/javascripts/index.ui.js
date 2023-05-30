(function (window, document, $, undefined) {
    $.extend({ index: {} });
    $.extend($.index, { var: {}, ui: {} });
    $.extend($.index.ui, {
        _init: function(){
            //const sess = new onnx.InferenceSession();
            //const loadingModelPromise = sess.loadModel("../weights/two_expert.onnx")
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
            $("#dropArea").on('change', '#image', function(){
                //console.log('-------');
                const file = this.files[0];
                print(file)
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

            // Read the image file as a data URL
            const reader = new FileReader();
            reader.onloadend = async function() {
                // Create an image element and load the data URL
                const imageElement = document.createElement('img');
                imageElement.src = reader.result;

                // Display the uploaded image
                $('#previewImg').attr('src', reader.result);
                $('#imagePreview').show();

                // Load the model and classify the image
                const model = await loadModel();
                const predictions = await classifyImage(model, imageElement);

                // Display the classification result
                const resultText = predictions.map(p => `${p.className}: ${p.probability.toFixed(4)}`).join('<br>');
                $('#classification').html(resultText);
            };
            reader.readAsDataURL(file);
        }
    })
    $(document).ready(function(){
        $.index.ui._init()
    });

}(window, document, jQuery))