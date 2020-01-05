import React, { Component } from 'react';
import './App.css';

import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import Image from 'react-bootstrap/Image'
// import Figure from 'react-bootstrap/Figure'
// import InputGroup from 'react-bootstrap/InputGroup'
// import FormControl from 'react-bootstrap/FormControl'

import 'bootstrap/dist/css/bootstrap.css';
import axios from 'axios'

// class ImageFigure extends Component {
//   constructor(props) {
//     super(props);
//     this.state = {
//       file: '',
//       imagePreviewUrl: '',
//       uploaded: false
//     };
//     this.handleImageChange = this.handleImageChange.bind(this);
//     this.handleSubmit = this.handleSubmit.bind(this);
//   }

//   // Handle user submitting image
//   handleSubmit(e) {
//     e.preventDefault();
//     // TODO: Handle user submitting file
//   }

//   // Handle new image
//   handleImageChange(e) {
//     e.preventDefault();

//     let reader = new FileReader();
//     let file = e.target.files[0];

//     reader.onloadend = () => {
//       this.setState({
//         file: file,
//         imagePreviewUrl: reader.result,
//         uploaded: true
//       });
//     }

//     reader.readAsDataURL(file);
//   }

//   // Render in react
//   render() {
//     let imagePreviewUrl = "https://i.ytimg.com/vi/MPV2METPeJU/maxresdefault.jpg";
//     let fileSelectText = this.state.uploaded ? this.state.file.name : "Choose file";
//     if (this.state.imagePreviewUrl !== '') {
//       imagePreviewUrl = this.state.imagePreviewUrl;
//     }

//     return (
//         <React.Fragment>
//           <Image
//             className = "mt-5"
//             width={200}
//             height={200}
//             alt="image-upload" 
//             src={imagePreviewUrl} 
//             thumbnail rounded/>
//           <div className="input-group mt-2 mb-2">
//           <div className="custom-file">
//             <input
//               type="file"
//               className="custom-file-input"
//               id="inputGroupFile01"
//               aria-describedby="inputGroupFileAddon01"
//               accept="image/*"
//               onChange={this.handleImageChange}
//             />
//             <label className="custom-file-label" htmlFor="inputGroupFile01">
//               {fileSelectText}
//             </label>
//           </div>
//         </div>
//       </React.Fragment>
//     );
//   }

// }

class App extends Component {

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
    if (this.state.file === '') {
      console.log("Error: No file selected");
      return
    }

    let url = 'http://localhost:4321/predict'

    let formData = new FormData();
    formData.append('image', this.state.file);
    axios.post(url, formData, {
      headers: {
        'content-type': 'multipart/form-data'
      }
    }).then((res) => {
      console.log(res.data);
    }).catch((err) => {
      console.log(err)
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
        uploaded: true
      });
    }

    reader.readAsDataURL(file);
  }

  // React render
  render() {
    let isLoading = false;
    let imagePreviewUrl = "https://i.ytimg.com/vi/MPV2METPeJU/maxresdefault.jpg";
    let fileSelectText = this.state.uploaded ? this.state.file.name : "Choose file";
    if (this.state.imagePreviewUrl !== '') {
      imagePreviewUrl = this.state.imagePreviewUrl;
    }
  
    return (
      <Container>
        <div>
          <h1 className="title">Dog Breed Classifier</h1>
        </div>
        <div className="content">
        <Form>
          <Row>
            <Col>
              <Image
              className = "mt-5"
              width={200}
              height={200}
              alt="image-upload" 
              src={imagePreviewUrl} 
              thumbnail rounded/>
              <div className="input-group mt-2 mb-2">
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
                onClick={!isLoading ? this.handleSubmit : null}>
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