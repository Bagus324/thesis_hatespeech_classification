document.querySelectorAll('.custom-file-upload input[type="file"]').forEach((input) => {
    input.addEventListener("change", function (event) {
        const fileName = event.target.files.length > 0 ? event.target.files[0].name : "Choose File";
        const span = event.target.nextElementSibling;
        span.textContent = fileName;
    });
});

$(document).ready(function () {
    $('#collapseContent').on('shown.bs.collapse', function () {
        $('.collapse-button')
            .find('i')
            .removeClass('fa-chevron-down')
            .addClass('fa-chevron-right');
        $('.collapse-button').removeClass('collapsed');
    });

    $('#collapseContent').on('hidden.bs.collapse', function () {
        $('.collapse-button')
            .find('i')
            .removeClass('fa-chevron-')
            .addClass('fa-chevron-down');
        $('.collapse-button').addClass('collapsed');
    });
});
