import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement } from 'chart.js';

ChartJS.register(Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement);

const FileUpload = () => {
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [anomalies, setAnomalies] = useState([]);
  const [dataset, setDataset] = useState([]);
  const [error, setError] = useState(null);
  const [highlightedAnomaly, setHighlightedAnomaly] = useState(null); // Store the highlighted anomaly index
  const anomaliesPerPage = 40; // Number of anomalies to show per page
  const [currentPage, setCurrentPage] = useState(1);

  // Handle file upload
  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log('Upload response:', response.data);

      setUploadedFileName(file.name); // Store the file name for anomaly detection

      if (response.data.anomalies) {
        const dataset = response.data.losses;  // Use `losses` or any column from the dataset
        setDataset(dataset); // Save dataset for graph rendering

        setAnomalies(response.data.anomalies);
        setError(null);  // Clear any previous errors
      } else {
        throw new Error('Invalid response from server. Anomalies data is missing.');
      }
    } catch (uploadError) {
      console.error('Error uploading file:', uploadError);
      setError('Error uploading file. Please try again.');
    }
  };

  // Detect anomalies
  const detectAnomalies = async () => {
    if (!uploadedFileName) {
      console.error('No uploaded file name found!');
      setError('No file uploaded. Please upload a file first.');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/api/detect-anomalies', {
        fileName: uploadedFileName, // Send the uploaded file name
      });

      if (response.data.anomalies) {
        console.log('Anomalies detected:', response.data.anomalies);
        setAnomalies(response.data.anomalies);
        setError(null); // Clear previous errors
      }
    } catch (anomalyError) {
      console.error('Error detecting anomalies:', anomalyError);
      setError('Error detecting anomalies. Please try again.');
    }
  };

  // Get the anomalies for the current page
  const indexOfLastAnomaly = currentPage * anomaliesPerPage;
  const indexOfFirstAnomaly = indexOfLastAnomaly - anomaliesPerPage;
  const currentAnomalies = anomalies.slice(indexOfFirstAnomaly, indexOfLastAnomaly);

  // Prepare chart data
  const chartData = {
    labels: dataset.map((_, index) => index),  // X-axis labels (index of data points)
    datasets: [
      {
        label: 'Dataset', // Normal data label
        data: dataset,    
        fill: false, // Line without filling under the graph
        borderColor: 'rgba(75,192,192,1)', // Blue color for normal points
        pointBackgroundColor: 'rgba(75,192,192,1)', // Normal data points color
        pointRadius: 3, // Thinner points for normal data
        borderWidth: 1, // Thinner lines for normal data
      },
      {
        label: 'Anomalies', // Anomalies label
        data: dataset.map((data, index) =>
          anomalies.includes(index) ? data : NaN // Only show anomalies, set other points as NaN
        ), 
        backgroundColor: 'red', // Red color for anomalies
        borderColor: 'red',
        pointRadius: 6, // Slightly thicker points for anomalies
        borderWidth: 2, // Thicker border for anomalies
        fill: false, // No filling for anomalies
        tension: 0 // No curve for this line
      },
      {
        label: 'Highlighted Anomaly', // Highlighted anomaly label
        data: highlightedAnomaly !== null ? [
          { x: highlightedAnomaly, y: dataset[highlightedAnomaly] } // Place anomaly at the right X and Y
        ] : [], // Only show the highlighted anomaly if it's clicked
        pointBackgroundColor: 'purple', // Purple color for highlighted anomaly
        pointRadius: 10, // Larger radius for highlighted anomaly
        borderColor: 'purple',
        fill: false, // No filling for highlighted anomaly
        tension: 0 // No curve for this point
      },
    ],
  };

  // Highlight the clicked anomaly
  const handleAnomalyClick = (index) => {
    setHighlightedAnomaly(index); // Set the highlighted anomaly index
  };

  // Configure the chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false, // Allow the chart to resize to fill container
    scales: {
      x: {
        type: 'linear', // Use linear scale for the X axis
        ticks: {
          maxRotation: 0, // Prevent X axis labels from rotating
          autoSkip: true,  // Skip labels to prevent overlap
        },
      },
      y: {
        ticks: {
          beginAtZero: false, // Don't force the Y-axis to start at zero (optional)
        },
      },
    },
  };

  // Handle page change
  const handlePageChange = (newPage) => {
    if (newPage > 0 && newPage <= Math.ceil(anomalies.length / anomaliesPerPage)) {
      setCurrentPage(newPage);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Anomaly Detection</h2>

      <div style={{ marginBottom: '15px' }}>
        <input
          type="file"
          onChange={(e) => handleFileUpload(e.target.files[0])}
          style={{ marginRight: '10px' }}
        />
        <button onClick={() => handleFileUpload(document.querySelector('input[type="file"]').files[0])}>
          Upload
        </button>
      </div>

      <button onClick={detectAnomalies} style={{ marginBottom: '15px' }}>
        Detect Anomalies
      </button>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {dataset.length > 0 && (
        <div style={{ width: '100%', height: '600px' }}>
          <Line data={chartData} options={chartOptions} />
        </div>
      )}

      {anomalies.length > 0 && (
        <div>
          <h3>Anomalies Detected</h3>

          {/* Display anomalies as clickable grid items */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
            {currentAnomalies.map((anomaly, index) => (
              <button
                key={index}
                onClick={() => handleAnomalyClick(anomaly)}
                style={{
                  textAlign: 'center',
                  padding: '5px',
                  border: '1px solid #ccc',
                  backgroundColor: highlightedAnomaly === anomaly ? 'yellow' : 'white',
                  cursor: 'pointer',
                  fontWeight: highlightedAnomaly === anomaly ? 'bold' : 'normal',
                }}
              >
                Index: {anomaly}
              </button>
            ))}
          </div>

          {/* Pagination controls */}
          <div>
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              Previous
            </button>
            <span>{` Page ${currentPage} of ${Math.ceil(anomalies.length / anomaliesPerPage)} `}</span>
            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === Math.ceil(anomalies.length / anomaliesPerPage)}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default frontend;
