import React, { Component } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import Image from 'react-bootstrap/Image'
import Figure from 'react-bootstrap/Figure'
import InputGroup from 'react-bootstrap/InputGroup'
import FormControl from 'react-bootstrap/FormControl'
import 'bootstrap/dist/css/bootstrap.css';

class ImageFigure extends Component {
  constructor(props) {
    super(props);
    this.state = {
      file: '',
      imagePreviewUrl: '',
      uploaded: false
    };
    this.handleImageChange = this.handleImageChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  // Handle user submitting image
  handleSubmit(e) {
    e.preventDefault();
    // TODO: Handle user submitting file
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
        uploaded: true
      });
    }

    console.log(file)

    reader.readAsDataURL(file);
  }

  // Render in react
  render() {
    let imagePreviewUrl = "https://i.ytimg.com/vi/MPV2METPeJU/maxresdefault.jpg";
    let fileSelectText = this.state.uploaded ? this.state.file.name : "Choose file";
    if (this.state.imagePreviewUrl !== '') {
      imagePreviewUrl = this.state.imagePreviewUrl;
    }

    return (
        <Figure>
          <Figure.Image alt="image-upload" src={imagePreviewUrl} thumbnail rounded height={299} width={299} />
          <div className="input-group">
          <div className="custom-file">
            <input
              type="file"
              className="custom-file-input"
              id="inputGroupFile01"
              aria-describedby="inputGroupFileAddon01"
              onChange={this.handleImageChange}
            />
            <label className="custom-file-label" htmlFor="inputGroupFile01">
              {fileSelectText}
            </label>
          </div>
        </div>
      </Figure>
    );
  }

}

class App extends Component {

  constructor(props) {
    super(props);

    this.state = {
      
    };
  }

  render() {
    let isLoading = false;
    let imgsrc = "https://i.ytimg.com/vi/MPV2METPeJU/maxresdefault.jpg";
  
    return (
      <Container>
        <div>
          <h1 className="title">Dog Breed Classifier</h1>
        </div>
        <div className="content">
        <Form>
          <Row>
            <Col>
              <ImageFigure />
            </Col>
            <Col>
              Graph goes here
            </Col>
          </Row>
          <Row>
            <Col>
              <Button
                block
                variant="success"
                disabled={isLoading}
                onClick={!isLoading ? this.handlePredictClick : null}>
                { isLoading ? 'Making prediction' : 'Predict' }
              </Button>
            </Col>
            <Col>
              <Button
                block
                variant="danger"
                disabled={isLoading}
                onClick={this.handleCancelClick}>
                Reset prediction
              </Button>
            </Col>
          </Row>
          </Form>
        </div>
      </Container>
    );
  }
}

export default App;