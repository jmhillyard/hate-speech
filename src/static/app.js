let dynamicTable = (function() {

  let _tableId, _table,
      _fields, _headers,
      _defaultText;

  /** Builds the row with columns from the specified names.
   *  If the item parameter is specified, the memebers of the names array will be used as property names of the item; otherwise they will be directly parsed as text.
   */
  function _buildRowColumns(names, item) {
      let row = '<tr>';
      if (names && names.length > 0)
      {
          $.each(names, function(index, name) {
              let c = item ? item[name+''] : name;
              row += '<td>' + c + '</td>';
          });
      }
      row += '<tr>';
      return row;
  }

  /** Builds and sets the headers of the table. */
  function _setHeaders() {
      // if no headers specified, we will use the fields as headers.
      _headers = (_headers == null || _headers.length < 1) ? _fields : _headers;
      let h = _buildRowColumns(_headers);
      if (_table.children('thead').length < 1) _table.prepend('<thead></thead>');
      _table.children('thead').html(h);
  }

  function _setNoItemsInfo() {
      if (_table.length < 1) return; //not configured.
      let colspan = _headers != null && _headers.length > 0 ?
          'colspan="' + _headers.length + '"' : '';
      let content = '<tr class="no-items"><td ' + colspan + ' style="text-align:center">' +
          _defaultText + '</td></tr>';
      if (_table.children('tbody').length > 0)
          _table.children('tbody').html(content);
      else _table.append('<tbody>' + content + '</tbody>');
  }

  function _removeNoItemsInfo() {
      let c = _table.children('tbody').children('tr');
      if (c.length == 1 && c.hasClass('no-items')) _table.children('tbody').empty();
  }

  return {
      /** Configres the dynamic table. */
      config: function(tableId, fields, headers, defaultText) {
          _tableId = tableId;
          _table = $('#' + tableId);
          _fields = fields || null;
          _headers = headers || null;
          _defaultText = defaultText || 'No items to list...';
          _setHeaders();
          _setNoItemsInfo();
          return this;
      },
      /** Loads the specified data to the table body. */
      load: function(data, append) {
          if (_table.length < 1) return; //not configured.
          _setHeaders();
          _removeNoItemsInfo();
          console.log(data)
          if (data && data.length > 0) {
              let rows = '';
              $.each(data, function(index, item) {
                  rows += _buildRowColumns(_fields, item);
              });
              let mthd = append ? 'append' : 'html';
              _table.children('tbody')[mthd](rows);
          }
          else {
              _setNoItemsInfo();
          }
          return this;
      },
      /** Clears the table body. */
      clear: function() {
          _setNoItemsInfo();
          return this;
      }
  };
}());

let get_database = function() {
$.ajax({
    url: '/load',
    contentType: "application/json; charset=utf-8",
    type: 'GET',
    success: function (data) {
      let dt = dynamicTable.config('prediction-table',
                               ['probability', 'label'],
                               ['Probability', 'Risk Label'], //set to null for field names instead of custom header names
                               'There are no items to list...');
        dt.load(JSON.parse('[' + data1 + ']'));
    },
})
};



$(document).ready(function(e) {

  let data1 = [
      { 'probability': '0.7', 'label': 'High', field3: 'value a3', field4: 'value a4' },
      { 'probability': '0.3', 'label': 'Low', field3: 'value b3', field4: 'value b4' },
      ];

  $('#btn-load').click(function(e) {
    data1//  get_database();
  });
  // setInterval(function() {document.getElementById("btn-load").click();}, 100);
  // setInterval(function(){$("btn-load").click();},2000);

});
