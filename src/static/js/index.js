function updateTableData() {
    $.ajax({
        url: '/table_data',
        type: 'GET',
        dataType: 'json',
        success: function(data) {
            $('#students tbody').empty();
            $.each(data, function(key, value) {
                var row = $('<tr>');
                row.append($('<td>').text(key));
                row.append($('<td>').text(value.name));
                row.append($('<td>').text(value.present ? 'Sim' : 'Não'));
                $('#students tbody').append(row);
            });
        }
    });
}
setInterval(updateTableData, 5000);
$(document).ready(function() {
    updateTableData();
});