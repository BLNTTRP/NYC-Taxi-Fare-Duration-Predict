import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def logout():
    st.session_state.logged_in = False
    st.rerun()

login_page = st.Page("pages/login.py", title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

create_user = st.Page("pages/create_user.py", title="Create New User", icon=":material/add_circle:")
main_page = st.Page("pages/main.py", title="Take Taxi", icon=":material/dashboard:", default=True)

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Take Taxi": [main_page]
        }
    )
else:
    pg = st.navigation([login_page, create_user])
    
pg.run()
