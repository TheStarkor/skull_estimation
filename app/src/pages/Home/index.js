
import { useState } from "react";

import { Button, Col, Row, Upload } from "antd";
import { PlusOutlined } from "@ant-design/icons";
import Chart from 'react-apexcharts'

import "./index.css"
import axios from "axios";

const Home = () => {
  const [isInfer, setInfer] = useState(false);
  const [isUploaded, setUpload] = useState(false);
  const [mastoidUrl, setMastoidUrl] = useState("");
  const [glabellaUrl, setGlabellaUrl] = useState("");
  const [supraorbitalUrl, setSupraorbitalUrl] = useState("");
  const [inferMessage, setInferMessage] = useState("");
  const [isLoading, setLoading] = useState(false);
  
  const [series, setSeries] = useState(
    [{
      name: 'Positive',
      data: [0.5, 0]
    },
    {
      name: 'Negative',
      data: [0, -0.8]
    }
    ]
  )
  const [options, setOptions] = useState({
      chart: {
        type: 'bar',
        height: 440,
        width: 1000,
        stacked: true,
        toolbar: {
          show: false
        }
      },
      colors: ['#008FFB', '#FF4560'],
      plotOptions: {
        bar: {
          horizontal: true,
          barHeight: '80%',
        },
      },
      dataLabels: {
        enabled: false
      },
      stroke: {
        width: 1,
        colors: ["#fff"]
      },
      legend: {
        show: false
      },
      grid: {
        xaxis: {
          lines: {
            show: false
          }
        }
      },
      yaxis: {
        min: -2,
        max: 2,
      },
      tooltip: {
        shared: false,
        x: {
          formatter: function (val) {
            return val
          }
        },
        y: {
          formatter: function (val) {
            return val
          }
        }
      },
      title: {
        text: inferMessage,
        align: 'center'
      },
      xaxis: {
        categories: ['남', '여'],
        title: {
          text: 'Gender Score'
        },
        labels: {
          formatter: function (val) {
            return Math.round(val)
          }
        }
      },
    }
  )

  const init = () => {
    setMastoidUrl("");
    setGlabellaUrl("");
    setSupraorbitalUrl("");
    setInfer(false);
    isUploaded(false);
  }

  const inference = async () => {
    setLoading(true);
    const res = await axios.post('https://9c95-143-248-107-187.ngrok.io/estimation', {
      mastoid: mastoidUrl,
      glabella: glabellaUrl,
      supraorbital: supraorbitalUrl
    });
    setLoading(false);
    console.log(res);

    if (res.data.gender === "1") {
      setInferMessage("여성일 확률이 높습니다.")
      setSeries([{
        name: 'Positive',
        data: [0, res.data.score_0]
      },
      {
        name: 'Negative',
        data: [res.data.score_1, 0]
      }
    ])
    } else {
      setInferMessage("남성일 확률이 높습니다.")
      setSeries([{
          name: 'Positive',
          data: [0, res.data.score_0]
        },
        {
          name: 'Negative',
          data: [res.data.score_1, 0]
        }
      ])
    }

    setInfer(true);
  }

  const moveToHome = () => {
    setInfer(false);
    isUploaded(false);
  }

  const isButtonDisable = () => {
    return (mastoidUrl === "") && (glabellaUrl === "") && (supraorbitalUrl === "")
  }

  const handleMastoidChange = (info) => {
    if (info.file.response) {
      setMastoidUrl(info.file.response.uri);
    }
  };

  const handleGlabellaChange = (info) => {
    if (info.file.response) {
      setGlabellaUrl(info.file.response.uri);
    }
  };

  const handleSupraorbitalChange = (info) => {
    if (info.file.response) {
      setSupraorbitalUrl(info.file.response.uri);
    }
  };

  
  const StartPage = () => {
    return (
      <>
        <h1>아래 예시에 해당하는</h1>
        <h1>이미지를 업로드해주세요</h1>

        <h3>세 부위 모두 업로드 할 시 정확도가 가장 높습니다.</h3>

        <Row>
          <Col span={12}>
            <img src={"https://tmi-image.s3.ap-northeast-2.amazonaws.com/profile/1655562218502skull-ear.png"} alt="avatar" style={{ width: "102px" }} />
            <h3>MASTOID</h3>
          </Col>
          <Col span={12}>
            <Upload
              name="image"
              listType="picture-card"
              className="avatar-uploader"
              showUploadList={false}
              action={
                "https://api.onebob.co/uploads"
              }
              onChange={handleMastoidChange}
            >
              {mastoidUrl ? (
                <img src={mastoidUrl} alt="avatar" style={{ width: "100%" }} />
              ) : (
                <div>
                  <PlusOutlined style={{ color: "#00000050" }} />
                </div>
              )}
            </Upload>
          </Col>
        </Row>

        <Row>
          <Col span={12}>
            <img src={"https://tmi-image.s3.ap-northeast-2.amazonaws.com/profile/1655562486949skull-eyeside.png"} alt="avatar" style={{ width: "102px" }} />
            <h3>GLABELLA</h3>
          </Col>
          <Col span={12}>
            <Upload
              name="image"
              listType="picture-card"
              className="avatar-uploader"
              showUploadList={false}
              action={
                "https://api.onebob.co/uploads"
              }
              onChange={handleGlabellaChange}
            >
              {glabellaUrl ? (
                <>
                  <img src={glabellaUrl} alt="avatar" style={{ width: "100%" }} />
                </>
              ) : (
                <div>
                  <PlusOutlined style={{ color: "#00000050" }} />
                  {/* <div style={{ marginTop: 8 , color:'#ffffff'}}>Profile</div> */}
                </div>
              )}
            </Upload>
          </Col>
        </Row>

        <Row>
          <Col span={12}>
            <img src={"https://tmi-image.s3.ap-northeast-2.amazonaws.com/profile/1655562473340skull-upview.png"} alt="avatar" style={{ width: "102px" }} />
            <h3>SUPRAORBITAL</h3>
          </Col>
          <Col span={12}>
            <Upload
              name="image"
              listType="picture-card"
              className="avatar-uploader"
              showUploadList={false}
              action={
                "https://api.onebob.co/uploads"
              }
              onChange={handleSupraorbitalChange}
            >
              {supraorbitalUrl ? (
                <img src={supraorbitalUrl} alt="avatar" style={{ width: "100%" }} />
              ) : (
                <div>
                  <PlusOutlined style={{ color: "#00000050" }} />
                </div>
              )}
            </Upload>
          </Col>
        </Row>

        <Button type="primary" onClick={inference} disabled={isButtonDisable()} style={{marginTop: '10px', width: '200px'}} loading={isLoading}>
          성별 예측하기
        </Button>
      </>
    )
  }

  const InferResultPage = () => {
    return (
      <>
        <Chart options={options} series={series} type="bar" height={250} />
        
        {mastoidUrl && 
          <>
            <Row>
              <Upload
                name="image"
                listType="picture-card"
                className="avatar-uploader"
                showUploadList={false}
                action={
                  "https://api.onebob.co/uploads"
                }
              >
                <img src={mastoidUrl} alt="avatar" style={{ width: "100%" }} />
              </Upload>
            </Row>
            <h3>MASTOID</h3>
          </>
        }

        {glabellaUrl &&
          <>
            <Row>
              <Upload
                name="image"
                listType="picture-card"
                className="avatar-uploader"
                showUploadList={false}
                action={
                  "https://api.onebob.co/uploads"
                }
              >
                <img src={glabellaUrl} alt="avatar" style={{ width: "100%" }} />
              </Upload>
            </Row>
            <h3>GLABELLA</h3>
          </>
        }
        
        {supraorbitalUrl &&
          <>
            <Row align="center">
              <Upload
                name="image"
                listType="picture-card"
                className="avatar-uploader"
                showUploadList={false}
                action={
                  "https://api.onebob.co/uploads"
                }
              >
                <img src={supraorbitalUrl} alt="avatar" style={{ width: "100%" }} />
              </Upload>
            </Row>
            <h3>SUPRAORBITAL</h3>
          </>
        }

        <Row>
          <Button type="primary" onClick={moveToHome} style={{marginTop: '10px', width: '200px'}}>
            뒤로가기
          </Button>
        </Row>

        <Row>
          <Button onClick={init} style={{marginTop: '10px', width: '200px'}}>
            처음으로
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