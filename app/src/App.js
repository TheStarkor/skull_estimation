import { BrowserRouter } from "react-router-dom";
import axios from "axios";

import "antd/dist/antd.css";

import Router from "./router";

function App() {
  axios.defaults.baseURL = "http://localhost:8000";
  // axios.interceptors.request.use((config) => {
  //   // TODO
  //   // if (process.env.REACT_APP_STAGE === 'dev') {
  //   //   config.baseURL = process.env.REACT_APP_DEV_BASE_URL
  //   // } else if (process.env.REACT_APP_STAGE === 'prod') {
  //   //   config.baseURL = process.env.REACT_APP_PROD_BASE_URL
  //   // }
  //   config = config || {};
  //   config.headers = config.headers || {};
  //   config.headers.Authorization = cookies.Authorization;

  //   return config;
  // });

  return (
    <BrowserRouter>
      <Router />
    </BrowserRouter>
  );
}

export default App;
