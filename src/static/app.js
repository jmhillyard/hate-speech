function drawChart() {
  // var data = myJSON
  var data = new google.visualization.arrayToDataTable(myJSON);
    // Declare columns
  //data.addRows(myJSON);
      console.log(myJSON);
      //console.log(array);
      console.log(data.getNumberOfRows());
  // var data = google.visualization.arrayToDataTable([
  //        ["Year","Sales","Expenses"],
  //        ['2004',1000,400],
  //        ['2005',1170,460],
  //        ['2006',660,1120],
  //        ['2007',1030,540]]);
  var options = {
    title: ' Negative Sentiment around Seacoast NH School Systems',
    curveType: 'function',
    legend: { position: 'bottom' }
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
