import React, { Component } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.css';

import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import Image from 'react-bootstrap/Image';
import Alert from 'react-bootstrap/Alert';

import Chart from 'react-apexcharts';
import axios from 'axios';


function Graph(props) {
  let labels = [];
  let probabilities = []
  props.predictions.forEach(item => {
    labels.push(item.label.split("_").map(w => w[0].toUpperCase() + w.substr(1)).join(" "))
    probabilities.push(item.probability)
  });
  console.log(labels);
  console.log(probabilities);

  let chartOptions = {
      chart: {
        type: 'bar',
        height: 350
      },
      plotOptions: {
        bar: {
          horizontal: true,
        }
      },
      dataLabels: {
        enabled: false
      },
      xaxis: {
        categories: labels,
      }
  };

  let chartSeries = [
    {data: probabilities}
  ];

  return (
    <div>
      <Chart options={chartOptions} series={chartSeries} type="bar"/>
    </div>);
}

class App extends Component {

  constructor(props) {
    super(props);
    this.state = {
      file: '',
      imagePreviewUrl: '',
      imageSelected: false,
      predictions: null,
      dog_found: false,
      isLoading: false,
    };
    this.handleImageChange = this.handleImageChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  // Handle user submitting image
  handleSubmit(e) {
    e.preventDefault();
    if (this.state.file === '') {
      console.log("Error: No file selected");
      return
    }

    // Prediction endpoint
    let url = 'http://localhost:4321/predict'

    let formData = new FormData();
    formData.append('image', this.state.file);
    this.setState({
      isLoading: true
    })
    axios.post(url, formData, {
      headers: {
        'content-type': 'multipart/form-data'
      }
    }).then((res) => {
      // console.log(res.data);
      this.setState({
        predictions: res.data.predictions,
        dogFound: res.data.dog_found,
        isLoading: false
      });
    }).catch((err) => {
      console.log(err);
      this.setState({
        isLoading: false
      });
    });
  }

  // Handle new image
  handleImageChange(e) {
    e.preventDefault();
    let reader = new FileReader();
    let file = e.target.files[0];

    reader.onloadend = () => {
      this.setState({
        file: file,
        imagePreviewUrl: reader.result,
        imageSelected: true
      });
    }
    reader.readAsDataURL(file);
  }

  // Handle chart
  chartHandler() {
    if (this.state.isLoading) {
      return ("loading...");
    } else if (this.state.predictions) {
      return this.state.dogFound ? (<Graph predictions={this.state.predictions}/>) : (<div/>);
    } else {
      return null;
    }
  }

  // Handle alert
  alertHandler() {
    let variant;
    let text;
    if (this.state.isLoading) {
      variant = "warning";
      text = "Loading...";
    } else if (this.state.predictions) {
      variant = this.state.dogFound ? "success" : "danger";
      text = this.state.dogFound ? "Dog found!" : "Dog not found :(";
    } else {
      return null;
    }

    return (<Alert variant={variant}>{text}</Alert>);
  }

  // React render
  render() {
    let imagePreviewUrl = "https://glowvarietyshow.com/wp-content/uploads/2017/03/placeholder-image.jpg";
    let fileSelectText = this.state.imageSelected ? this.state.file.name : "Choose file";
    if (this.state.imagePreviewUrl !== '') {
      imagePreviewUrl = this.state.imagePreviewUrl;
    }

    let chartElement = this.chartHandler();
    let alertElement = this.alertHandler();
  
    return (
      <Container>
        <div>
          <h1 className="title">Dog Breed Classifier</h1>
        </div>
        <div className="content">
          <Row>
            <Col sm={4} md={4}> 
              <Image
                className = "mt-2"
                width={200}
                height={200}
                alt="image-upload" 
                src={imagePreviewUrl} 
                thumbnail rounded/>
              <div className="input-group mt-3 mb-3">
                <div className="custom-file">
                  <input
                    type="file"
                    className="custom-file-input"
                    id="inputGroupFile01"
                    aria-describedby="inputGroupFileAddon01"
                    accept="image/*"
                    onChange={this.handleImageChange}
                  />
                  <label className="custom-file-label" htmlFor="inputGroupFile01">
                    {fileSelectText}
                  </label>
                </div>
              </div>
            </Col>
            <Col sm={8} md={8}>
              {chartElement}
            </Col>
          </Row>
          <Row>
            <Col sm={4} md={4}>
              <Button
                block
                variant="outline-success"
                disabled={this.state.isLoading}
                onClick={!this.state.isLoading ? this.handleSubmit : null}>
                {this.state.isLoading ? 'Making prediction' : 'Predict' }
              </Button>
            </Col>
            <Col sm={8} md={8}>
             {alertElement}
            </Col>
          </Row>
        </div>
      </Container>
    );
  }
}

export default App;