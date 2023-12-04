$(document).ready(function() {
    $('#predictionForm').submit(function(e) {
        e.preventDefault();

        // Perform prediction logic (replace with your actual prediction logic)
        const prediction = Math.random() < 0.5 ? 'Fake' : 'True';
        const probabilities = {
            'True': Math.random().toFixed(4),
            'Fake': Math.random().toFixed(4)
        };

        // Update result elements
        $('#prediction').text(prediction);
        $('#probTrue').text(probabilities['True']);
        $('#probFake').text(probabilities['Fake']);

        // Show the result div
        $('#result').removeClass('d-none');
        $('#probability').removeClass('d-none');
    });
});

