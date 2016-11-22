function drawChart() {
  // var data = myJSON
  var data = new google.visualization.arrayToDataTable(myJSON);
  //console.log(myJSON)
  var options = {
   title: '',
    curveType: 'none',
    series: {
            0: { color: '#1c91c0' },
            1: { color: '#e2431e' },
            2: { color: '#f1ca3a' }
          },
    legend: { position: 'center' }
  };

  var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));

  chart.draw(data, options);

}

function myFunc(vars) {
  return vars
}

function decodeJSON(encodedJSON) {
            var decodedJSON = $('<div/>').html(encodedJSON).text();
            return $.parseJSON(decodedJSON);
}
