
import { useState } from "react";

import { Button, Row, Upload } from "antd";
import { PlusOutlined } from "@ant-design/icons";

import "./index.css"

const Home = () => {
  const [isInfer, setInfer] = useState(false);
  const [isUploaded, setUpload] = useState(false);
  const [fileList, setFileList] = useState([
    {
      uid: '-1',
      name: 'image.png',
      status: 'done',
      url: 'https://zos.alipayobjects.com/rmsportal/jkjgkEfvpUPVyRjUImniVslZfWPnJuuZ.png',
    },
    {
      uid: '-2',
      name: 'image.png',
      status: 'done',
      url: 'https://zos.alipayobjects.com/rmsportal/jkjgkEfvpUPVyRjUImniVslZfWPnJuuZ.png',
    },
  ])


  const init = () => {
    setFileList([]);
    setInfer(false);
    isUploaded(false);
  }

  const inference = () => {
    setInfer(true);
  }

  const moveToHome = () => {
    setInfer(false);
    isUploaded(false);
  }

  const handleChange = ({ fileList: newFileList }) => {
    setFileList(newFileList);
    setUpload(true);
  }
  
  const StartPage = () => {
    return (
      <>
        <h1>이미지를 업로드 해주세요</h1>

        <Row>
          <Upload
            name="image"
            listType="picture-card"
            className="avatar-uploader"
            fileList={fileList}
            action={
              "https://api.onebob.co/uploads"
            }
            onChange={handleChange}
          >
            {fileList.length >= 5 ? null : <PlusOutlined style={{ color: "#00000050" }} />}
          </Upload>
        </Row>

        {isUploaded && 
          <Button type="primary" onClick={inference}>
            성별 예측하기
          </Button>
        }
      </>
    )
  }

  const InferResultPage = () => {
    return (
      <>
        <h1>97% 남성입니다.</h1>
        <Row>
          <Upload
            name="image"
            listType="picture-card"
            className="avatar-uploader"
            fileList={fileList}
            action={
              "https://api.onebob.co/uploads"
            }
            onChange={handleChange}
          >
          </Upload>
        </Row>

        <Row>
          <Button type="primary" onClick={moveToHome}>
            이미지 추가하기
          </Button>
        </Row>

        <Row>
          <Button type="primary" onClick={init}>
            다른 유골 측정하기
          </Button>
        </Row>
      </>
    )
  }

  return (
    <>
      <div className="home-container">
        {isInfer ? InferResultPage() : StartPage()}
      </div>
    </>
  )
}

export default Home;