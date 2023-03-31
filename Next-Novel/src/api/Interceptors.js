import axios from 'axios'
import { useContext, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { AuthContext } from "../context/AuthContext"


const instance = axios.create({
  baseURL : 'http://localhost:8000/api/'
})

const tokeninstance = axios.create({
  baseURL : 'http://localhost:8000/api/'
})

// token 필요없는 api
instance.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    console.log(error)
    return Promise.reject(error)
  }
)

instance.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    return Promise.reject(error)
  }
)

// function TokenInterceptor({children}) {

//   useEffect(() => {
//     console.log("@@@")
//     const resInterceptor = response => {
//       return response
//     }

//     const errInterceptor = error => {
//       return Promise.reject(error)
//     }

//     const interceptor = tokeninstance.interceptors.response.use(resInterceptor, errInterceptor)
//     return () => tokeninstance.interceptors.response.eject(interceptor)
//   }, [])
//   return children
// }


// 토큰 필요한 api
tokeninstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token')
    // const token = user.access_token
    if( token ) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    console.log(error)
    return Promise.reject(error)
  }
)

tokeninstance.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    return Promise.reject(error)
  }
)


export { instance, tokeninstance }