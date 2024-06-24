$(document).ready(function() {
    // Extend DataTables filtering
    $.fn.dataTable.ext.search.push(
        function(settings, data, dataIndex) {
            var min = $('#start').val();
            var max = $('#end').val();
            var date = data[2]; // Use data for the date column

            if (min && !isNaN(new Date(min).getTime())) {
                min = new Date(min);
            } else {
                min = null;
            }

            if (max && !isNaN(new Date(max).getTime())) {
                max = new Date(max);
            } else {
                max = null;
            }

            if (date && !isNaN(new Date(date).getTime())) {
                date = new Date(date);
            } else {
                return false;
            }

            if ((min === null && max === null) ||
                (min === null && date <= max) ||
                (min <= date && max === null) ||
                (min <= date && date <= max)) {
                return true;
            }
            return false;
        }
    );

    // Initialize DataTable with Buttons for exporting
    var table = $('#dataTable').DataTable({
        "order": [[ 2, "desc" ]],
        "columnDefs": [
            { "width": "45%", "targets": 0 },
            { "width": "30%", "targets": 1 },
            { "width": "25%", "targets": 2 }
        ],
        dom: 'Bfrtip', // Include Buttons extension
        buttons: [
            {
                extend: 'csvHtml5',
                text: 'Download CSV',
                className: 'download-button',
                title: function() {
                    var start = $('#start').val();
                    var end = $('#end').val();
                    var title = 'Data - ';
                    if (start && end) {
                        title += start + ' to ' + end;
                    } else if (start) {
                        title += start + ' to now';
                    } else if (end) {
                        title += 'ALL to ' + end;
                    } else {
                        title += 'ALL';
                    }
                    return title;
                },
                customize: function (csv) {
                    return csv;
                }
            },
            {
                extend: 'pdfHtml5',
                text: 'Download PDF',
                className: 'download-button',
                title: function() {
                    var start = $('#start').val();
                    var end = $('#end').val();
                    var title = 'Data - ';
                    if (start && end) {
                        title += start + ' to ' + end;
                    } else if (start) {
                        title += start + ' to now';
                    } else if (end) {
                        title += 'ALL to ' + end;
                    } else {
                        title += 'ALL';
                    }
                    return title;
                },
                orientation: 'landscape', // PDF orientation
                pageSize: 'A4' // PDF page size
            }
        ],
        initComplete: function() {
            $('.download-button').css({
                'background-color': 'white',
                'color': 'black',
                'border': '3px solid #ababab',
                'padding': '10px 20px',
                'text-align': 'center',
                'text-decoration': 'none',
                'display': 'inline-block',
                'font-size': '16px',
                'margin': '4px 2px',
                'cursor': 'pointer',
                'border-radius': '8px',
                'transition-duration': '0.4s'
            }).hover(
                function() {
                    $(this).css('background-color', '#ababab'); // Hover color
                },
                function() {
                    $(this).css('background-color', 'white'); // Original color
                }
            );


            $('#dataTable').css({
                'border-collapse': 'collapse',
                'width': '100%',
                'border': '1px solid #ddd',
                'font-size': '18px'
            });

            $('#dataTable th, #dataTable td').css({
                'text-align': 'left',
                'padding': '12px'
            });

            $('#dataTable tr').hover(function() {
                $(this).css('background-color', '#f1f1f1');
            }, function() {
                $(this).css('background-color', '');
            });

            $('#dataTable th').css({
                'background-color': '#4b4b4b',
                'color': 'white'
            });
            $('#dataTable tbody').css({
                'color': 'black'
            });

        }
    });

    // Event listener for the date inputs
    $('#start, #end').change(function() {
        table.draw();
    });
});
