#include <mitsuba/render/scene.h>
#include "myOM.h"

MTS_NAMESPACE_BEGIN

class TestOMIntegrater : public SamplingIntegrator
{
public:
    MTS_DECLARE_CLASS()
public:
    /// Initialize the integrator with the specified properties
    TestOMIntegrater(const Properties &props) : SamplingIntegrator(props) {
        Spectrum defaultColor;
        // defaultColor.fromLinearRGB(0.2f, 0.5f, 0.2f);
        defaultColor.fromLinearRGB(1.f, 1.f, 1.f);
        m_color = props.getSpectrum("color", defaultColor);
    }

private:
    Spectrum m_color;
    Float m_maxDist;
    AABB m_baseAABB;

    OM m_om;
    OM test_om;
    OM roma[OMNUM];

public:
    /// Unserialize from a binary data stream
    TestOMIntegrater(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}

    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int cameraResID,
                    int samplerResID)
    {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
                                       cameraResID, samplerResID);

        Point m_min(1e30f), m_max(-1e30f), m_center, m_lcorner;
        auto meshes = scene->getMeshes();

        for (auto m : meshes)
        {
            // SLog(EInfo, m->toString().c_str());
            m_min.x = std::min(m_min.x, m->getAABB().min.x);
            m_min.y = std::min(m_min.y, m->getAABB().min.y);
            m_min.z = std::min(m_min.z, m->getAABB().min.z);
            m_max.x = std::max(m_max.x, m->getAABB().max.x);
            m_max.y = std::max(m_max.y, m->getAABB().max.y);
            m_max.z = std::max(m_max.z, m->getAABB().max.z);
        }

        Vector3 d = m_max - m_min;
        Float r = d.length();
        r *= 0.5 * 1.001f;
        m_center = m_min + d / 2.0f;
        m_lcorner = m_center - Vector3(r);
        m_baseAABB = AABB(m_lcorner, m_lcorner + Vector3(2 * r));

        m_om.clear();
        m_om.setAABB(m_baseAABB);
        m_om.setSize(2 * r);
        m_om.setScene(scene);

        /* init roma */
        for (int i = 0; i < OMNUMSQRT; i++)
            for (int j = 0; j < OMNUMSQRT; j++)
            {
                roma[i * OMNUMSQRT + j].clear();
                roma[i * OMNUMSQRT + j].setAABB(m_baseAABB);
                roma[i * OMNUMSQRT + j].setSize(2 * r);
                m_om.generateROMA(&roma[i * OMNUMSQRT + j], Point2((i + 0.5f) / OMNUMSQRT, (j + 0.5f) / OMNUMSQRT));
            }
        test_om.clear();
        test_om.setAABB(m_baseAABB);
        test_om.setSize(2 * r);
        m_om.generateROMA(&test_om, Point2(0.5, 0.5));

        /* Find the camera position at t=0 seconds */
        Point cameraPosition = scene->getSensor()->getWorldTransform()->eval(0).transformAffine(Point(0.0f));
        m_maxDist = -std::numeric_limits<Float>::infinity();

        for (int i = 0; i < 8; ++i)
            m_maxDist = std::max(m_maxDist,
                                 (cameraPosition - scene->getAABB().getCorner(i)).length());

        std::ostringstream oss;
        oss << "Meshes: " << meshes.size() << std::endl;
        oss << m_om.toString() << std::endl;
        oss << "Max distance: " << m_maxDist << std::endl;
        oss << cameraPosition.toString() << std::endl;
        SLog(EDebug, oss.str().c_str());

        return true;
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        Float nearT;

        Ray r2(r);
        r2.d = normalize(Vector(1,1,1));
        int id = OM::nearestOMindex(r2);
        // return Spectrum(Float(id) / OMNUM) * m_color;
        // SLog(EInfo, "id: %d", id);
        if (roma[id].rayIntersect(r, nearT))
        {
            return Spectrum(1.01f - nearT / m_maxDist) * m_color;
        }

        // if (rRec.rayIntersect(r)) {
        //     Float distance = rRec.its.t;
        //     return Spectrum(1.0f - distance/m_maxDist) * m_color;
        // }
        return Spectrum(0.0f);
    }
};

MTS_IMPLEMENT_CLASS_S(TestOMIntegrater, false, SamplingIntegrator)

MTS_EXPORT_PLUGIN(TestOMIntegrater, "Occupancy map test");

MTS_NAMESPACE_END